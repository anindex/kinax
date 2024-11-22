from typing import Tuple, Any

from brax import math
from brax import scan
from brax.base import Motion
from brax.base import Transform
import jax
from jax import numpy as jnp

from kinax.model import URDFSystem


def forward(
    sys: URDFSystem, q: jnp.array, qd: jnp.array
) -> Tuple[Transform, Motion]:
    # set the input q and qd
    brax_q = jnp.zeros(sys.q_size())
    brax_qd = jnp.zeros(sys.qd_size())
    q = brax_q.at[sys.joint_ids].set(q)
    qd = brax_qd.at[sys.joint_ids].set(qd)
    # forward kinematics
    def jcalc(typ, q, qd, motion):
        if typ == 'f':
            j = Transform(pos=q[0:3], rot=q[3:7])
            jd = Motion(ang=qd[3:6], vel=qd[0:3])
        else:
            rot_fn = lambda ang, q: math.normalize(math.quat_rot_axis(ang, q))[0]
            j = Transform.create(
                rot=jax.vmap(rot_fn)(motion.ang, q),
                pos=jax.vmap(jnp.multiply)(motion.vel, q),
            )
            jd = jax.vmap(lambda a, b: a * b)(motion, qd)

            num_links, num_dofs = qd.shape[0] // int(typ), int(typ)
            s = (num_links, num_dofs, -1)
            j_stack, jd_stack = j.reshape(s), jd.reshape(s)

            j, jd = j_stack.take(0, axis=1), jd_stack.take(0, axis=1)
            for i in range(1, num_dofs):
                j_i, jd_i = j_stack.take(i, axis=1), jd_stack.take(i, axis=1)
                j = j.vmap().do(j_i)

                jd = jd + Motion(
                    ang=jax.vmap(math.rotate)(jd_i.ang, j_i.rot),
                    vel=jax.vmap(math.rotate)(
                        jd_i.vel + jax.vmap(jnp.cross)(j_i.pos, jd_i.ang), j_i.rot
                    ),
                )

        return j, jd

    j, jd = scan.link_types(sys, jcalc, 'qdd', 'l', q, qd, sys.dof.motion)

    j = sys.link.joint.vmap().do(j)  # joint transform
    j = sys.link.transform.vmap().do(j)  # link transform

    def world(parent, j, jd):
        """Convert transform/motion from joint frame to world frame."""
        if parent is None:
            jd = jd.replace(ang=jax.vmap(math.rotate)(jd.ang, j.rot))
            return j, jd
        x_p, xd_p = parent
        x = x_p.vmap().do(j)
        vel = xd_p.vel + jax.vmap(jnp.cross)(xd_p.ang, x.pos - x_p.pos)
        vel += jax.vmap(math.rotate)(jd.vel, x_p.rot)
        ang = xd_p.ang + jax.vmap(math.rotate)(jd.ang, x.rot)
        xd = Motion(vel=vel, ang=ang)
        return x, xd

    x, xd = scan.tree(sys, world, 'll', j, jd)

    x = x.replace(rot=jax.vmap(math.normalize)(x.rot)[0])

    return x, xd
