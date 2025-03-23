from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from flax import struct
from jax import numpy as jnp
import jax
from brax.base import DoF, Link, Actuator
from kinax.utils.files import get_robot_path


Q_WIDTHS = {'f': 7, '1': 1, '2': 2, '3': 3}
QD_WIDTHS = {'f': 6, '1': 1, '2': 2, '3': 3}


@struct.dataclass
class URDFSystem:

    dt: jax.Array
    gravity: jax.Array
    link: Link
    dof: DoF

    link_names: Tuple[str] = struct.field(pytree_node=False)
    link_types: str = struct.field(pytree_node=False)
    link_parents: Tuple[int, ...] = struct.field(pytree_node=False)
    joint_ids: jax.Array = struct.field(pytree_node=False)
    init_q: jax.Array = struct.field(pytree_node=False, default=None)

    def num_links(self) -> int:
        """Returns the number of links in the system."""
        return len(self.link_types)

    def dof_link(self, depth=False) -> jax.Array:
        """Returns the link index corresponding to each system dof."""
        link_idxs = []
        for i, link_type in enumerate(self.link_types):
            link_idxs.extend([i] * QD_WIDTHS[link_type])
        if depth:
            depth_fn = lambda i, p=self.link_parents: p[i] + 1 and 1 + depth_fn(p[i])
        depth_count = {}
        link_idx_depth = []
        for i in range(self.num_links()):
            depth = depth_fn(i)
            depth_idx = depth_count.get(depth, 0)
            depth_count[depth] = depth_idx + 1
            link_idx_depth.append(depth_idx)
        link_idxs = [link_idx_depth[i] for i in link_idxs]

        return jnp.array(link_idxs)

    def dof_ranges(self) -> List[List[int]]:
        """Returns the dof ranges corresponding to each link."""
        beg, ranges = 0, []
        for t in self.link_types:
            ranges.append(list(range(beg, beg + QD_WIDTHS[t])))
        beg += QD_WIDTHS[t]
        return ranges

    def q_idx(self, link_type: str) -> jax.Array:
        """Returns the q indices corresponding to a link type."""
        idx, idxs = 0, []
        for typ in self.link_types:
            if typ in link_type:
                idxs.extend(range(idx, idx + Q_WIDTHS[typ]))
        idx += Q_WIDTHS[typ]
        return jnp.array(idxs)

    def qd_idx(self, link_type: str) -> jax.Array:
        """Returns the qd indices corresponding to a link type."""
        idx, idxs = 0, []
        for typ in self.link_types:
            if typ in link_type:
                idxs.extend(range(idx, idx + QD_WIDTHS[typ]))
        idx += QD_WIDTHS[typ]
        return jnp.array(idxs)

    def q_size(self) -> int:
        """Returns the size of the q vector (joint position) for this system."""
        return sum([Q_WIDTHS[t] for t in self.link_types])

    def qd_size(self) -> int:
        """Returns the size of the qd vector (joint velocity) for this system."""
        return sum([QD_WIDTHS[t] for t in self.link_types])


# path names for the URDF file
FRANKA_PANDA_NO_GRIPPER = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_no_gripper.urdf'
FRANKA_PANDA = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_hand.urdf'
KUKA_IIWA7 = get_robot_path() / 'kuka_iiwa' / 'urdf' / 'iiwa7.urdf'
UR5 = get_robot_path() / 'ur_description' / 'urdf' / 'ur5.urdf'
UR10 = get_robot_path() / 'ur_description' / 'urdf' / 'ur10.urdf'
HABITAT_STRETCH = get_robot_path() / 'habitat_stretch' / 'urdf' / 'hab_stretch.urdf'
TIAGO_DUAL_HOLO = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal.urdf'
TIAGO_DUAL_HOLO_MOVE = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal_holonomic.urdf'
TIAGO_DUAL_WHEEL = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_wheeled_minimal.urdf'
SHADOW_HAND = get_robot_path() / 'shadow_hand' / 'shadow_hand.urdf'
ALLEGRO_HAND = get_robot_path() / 'allegro_hand' / 'allegro_hand.urdf'
PLANAR_2_LINK = get_robot_path() / 'planar_manipulators' / 'urdf' / '2_link_planar.urdf'
BAXTER = get_robot_path() / 'baxter' / 'baxter_spherized.urdf'
