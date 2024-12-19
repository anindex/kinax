from urdf_parser_py.urdf import URDF

from brax.base import (
    DoF,
    Inertia,
    Link,
    Motion,
    Transform,
)
import jax
from jax import numpy as jnp
import numpy as np

from kinax.model import URDFSystem


def euler_to_quat(v: jnp.array) -> jnp.array:
    c1, c2, c3 = jnp.cos(v / 2)
    s1, s2, s3 = jnp.sin(v / 2)
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * c2 * c3 + c1 * s2 * s3
    y = c1 * s2 * c3 - s1 * c2 * s3
    z = c1 * c2 * s3 + s1 * s2 * c3
    return jnp.asarray([w, x, y, z])
 

def load_model(model_path: str) -> URDFSystem:
    """
    Creates a brax system from a URDF file.
    https://wiki.ros.org/urdf/XML
    """
    urdf_model = URDF.from_xml_file(model_path)
    links = urdf_model.links
    joints = urdf_model.joints
    # transmissions = urdf_model.transmissions

    body_ipos = []
    body_iquat = []
    body_inertia = []
    body_mass = []
    link_id_map = {}

    joint_pos = [np.asarray([0.0, 0.0, 0.0])]  # NOTE: dummy root joint
    joint_quat = [np.asarray([1.0, 0.0, 0.0, 0.0])]  # NOTE: dummy root joint
    joint_types = []
    joint_axis = []
    joint_limit_efforts = []
    joint_limits = []
    joint_limit_velocities = []
    joint_damping = []
    joint_friction = []

    for i, link in enumerate(links):
        link_id_map[link.name] = i
        if link.inertial is not None and link.inertial.origin is not None:
            body_ipos.append(np.asarray(link.inertial.origin.xyz))
            body_iquat.append(euler_to_quat(np.asarray(link.inertial.origin.rpy)))
            body_inertia.append(np.asarray(link.inertial.inertia.to_matrix()))
            body_mass.append([link.inertial.mass])
        else:
            print(f'No inertial data for link: {link.name}, using default values!')
            body_ipos.append(np.asarray([0., 0., 0.]))
            body_iquat.append(np.asarray([1., 0., 0., 0.]))
            body_inertia.append(np.eye(3))
            body_mass.append([1.])

    for joint in joints:
        if joint.origin is None:
            joint_pos.append(np.zeros(3))
            joint_quat.append(np.asarray([1.0, 0.0, 0.0, 0.0]))
        else:
            joint_pos.append(np.asarray(joint.origin.xyz))
            rpy = np.asarray(joint.origin.rpy)
            joint_quat.append(euler_to_quat(rpy))
        joint_types.append(joint.joint_type)
        if joint.axis is None:
            joint_axis.append(np.zeros(3))
        else:
            joint_axis.append(np.asarray(joint.axis))
        if joint.type == 'fixed':
            joint_limit_efforts.append([0.])
            joint_limits.append([0., 0.])
            joint_limit_velocities.append([0.])
        else:
            if joint.limit is None:
                joint_limit_efforts.append([0.])
                joint_limits.append([-np.inf, np.inf])
                joint_limit_velocities.append([np.inf])
            else:
                joint_limit_efforts.append([joint.limit.effort])
                joint_limits.append([joint.limit.lower, joint.limit.upper])
                joint_limit_velocities.append([joint.limit.velocity])
        if joint.dynamics is None:
            joint_damping.append([1.])
            joint_friction.append([1.])
        else:
            joint_damping.append([joint.dynamics.damping])
            joint_friction.append([joint.dynamics.friction])

    body_ipos = np.asarray(body_ipos)
    body_iquat = np.asarray(body_iquat)
    body_inertia = np.asarray(body_inertia)
    body_mass = np.asarray(body_mass)

    joint_pos = np.asarray(joint_pos)
    joint_quat = np.asarray(joint_quat)
    joint_axis = np.asarray(joint_axis)
    joint_limit_efforts = np.asarray(joint_limit_efforts)
    joint_limits = np.asarray(joint_limits)
    joint_limit_velocities = np.asarray(joint_limit_velocities)
    joint_damping = np.asarray(joint_damping)
    joint_friction = np.asarray(joint_friction)

    # construct links object
    # identity = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (body_ipos.shape[0], 1))
    brax_link = Link(
        transform=Transform(pos=body_ipos, rot=body_iquat),  # NOTE: link pose is the same as inertial pose
        inertia=Inertia(  # pytype: disable=wrong-arg-types  # jax-ndarray
            transform=Transform(pos=body_ipos, rot=body_iquat),  # pytype: disable=wrong-arg-types  # jax-ndarray
            i=body_inertia,
            mass=body_mass,
        ),
        invweight=np.ones((body_ipos.shape[0], )),
        joint=Transform(pos=joint_pos, rot=joint_quat),  # pytype: disable=wrong-arg-types  # jax-ndarray
        constraint_stiffness=np.ones((body_ipos.shape[0], )),
        constraint_vel_damping=joint_damping,
        constraint_limit_stiffness=np.ones((body_ipos.shape[0], )),
        constraint_ang_damping=joint_damping,
    )

    brax_link = jax.tree_map(lambda x: x, brax_link)

    # construct DOF object
    # NOTE: always assume fixed root joint for now
    motions = [Motion(ang=np.zeros((1, 3)), vel=np.zeros((1, 3)))]
    limits = [(np.asarray([0.]), np.asarray([0.]))]
    joint_type_str = '1'
    joint_ids = []
    for i, typ in enumerate(joint_types):
        if typ == 'floating':
            motion = Motion(ang=np.eye(6, 3, -3), vel=np.eye(6, 3))
            limit = np.asarray([-np.inf] * 6), np.asarray([np.inf] * 6)
            joint_ids.append(i + 1)
            joint_type_str += 'f'
        elif typ == 'revolute':
            motion = Motion(ang=joint_axis[[i], :], vel=np.zeros((1, 3)))
            limit = joint_limits[i][0:1], joint_limits[i][1:2]
            joint_ids.append(i + 1)
            joint_type_str += '1'
        elif typ == 'continuous':
            motion = Motion(ang=joint_axis[[i], :], vel=np.zeros((1, 3)))
            limit = np.asarray([-np.inf]), np.asarray([np.inf])
            joint_ids.append(i + 1)
            joint_type_str += '1'
        elif typ == 'prismatic':
            motion = Motion(ang=np.zeros((1, 3)), vel=joint_axis[[i], :])
            limit = joint_limits[i][0:1], joint_limits[i][1:2]
            joint_ids.append(i + 1)
            joint_type_str += '1'
        elif typ == 'fixed':
            motion = Motion(ang=np.zeros((1, 3)), vel=np.zeros((1, 3)))
            limit = np.asarray([0.]), np.asarray([0.])
            joint_type_str += '1'  # NOTE: although fixed joint is not a DoF, we still need to add a dummy DoF to keep the joint index consistent
        else:
            raise ValueError(f'Joint type {typ} not supported!')
        motions.append(motion)
        limits.append(limit)

    motion = jax.tree_map(lambda *x: np.concatenate(x), *motions)
    limit = jax.tree_map(lambda *x: np.concatenate(x), *limits)
    brax_dof = DoF(
        motion=motion,
        armature=None,
        stiffness=None,
        damping=joint_damping,
        limit=limit,
        invweight=np.ones((body_ipos.shape[0], )),
        solver_params=None,
    )

    # construct link geomertries
    # TODO: resolve geom of link with Mesh, Box, Capsule, Cylinder, Sphere

    # construct actuator object
    # TODO: resolve Actuator with URDF transmission for forward dynamic

    # link names
    link_names = tuple(link.name for link in links)

    # link types
    link_types = joint_type_str  # NOTE: since link-joint is coupled in URDF, we use joint type to represent link type
    
    # link parents
    link_parents = [-1]  # NOTE: root link has no parent
    link_parents.extend([link_id_map[joint.parent] for joint in joints])

    sys = URDFSystem(
        dt=0.01,
        gravity=np.asarray([0., 0., -9.81]),
        link=brax_link,
        dof=brax_dof,
        link_names=link_names,
        link_types=link_types,
        link_parents=link_parents,
        joint_ids=np.asarray(joint_ids),
    )

    sys = jax.tree_map(jnp.asarray, sys)

    return sys
