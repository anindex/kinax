
import jax
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np

import matplotlib.pyplot as plt
import time
from brax.kinematics import world_to_joint, inverse

import kinax
from kinax.model import FRANKA_PANDA_NO_GRIPPER, FRANKA_PANDA, UR10, UR10_SUCTION, TIAGO_DUAL_HOLO, SHADOW_HAND, ALLEGRO_HAND, PLANAR_2_LINK, BAXTER
from kinax.skeleton import get_skeleton_from_system
from kinax.kinematics import forward


if __name__ == "__main__":

    sys = kinax.load_model(BAXTER)
    print('link_names: ', sys.link_names)
    print('link_parents: ', sys.link_parents)
    print('joint_ids: ', sys.joint_ids)
    print('joint_limits', jnp.stack(sys.dof.limit).T[sys.joint_ids])

    rng = jax.random.PRNGKey(0)
    q = jax.random.uniform(rng, shape=(len(sys.joint_ids),))
    # q = jnp.array([0.012, -0.57, 0., -2.81 , 0., 3.037, 0.741])
    rng, rng2 = jax.random.split(rng)
    qd = jnp.zeros_like(q)

    tic = time.time()
    x, xd = jit(forward)(sys, q, qd)
    toc = time.time()
    print(f'FK JIT compilation takes: {toc - tic}s')

    tic = time.time()
    x, xd = jit(forward)(sys, q, qd)
    toc = time.time()
    print(f'FK takes: {toc - tic}s')

    j, jd, _, _ = world_to_joint(sys, x, xd)

    tic = time.time()
    q_ik, qd_ik = jit(inverse)(sys, j, jd)
    toc = time.time()
    print(f'IK JIT compilation takes: {toc - tic}s')

    tic = time.time()
    q_ik, qd_ik = jit(inverse)(sys, j, jd)
    toc = time.time()
    print(f'IK takes: {toc - tic}s')

    print(f'q closed: {jnp.allclose(q, q_ik[sys.joint_ids], atol=1e-3)}')
    print(f'qd closed: {jnp.allclose(qd, qd_ik[sys.joint_ids], atol=1e-3)}')

    skeleton = get_skeleton_from_system(sys, q, qd)
    skeleton_ik = get_skeleton_from_system(sys, q_ik[sys.joint_ids], qd_ik[sys.joint_ids])
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
    skeleton.draw_skeleton(ax=ax)
    skeleton_ik.draw_skeleton(ax=ax, c='r')
    plt.show()
