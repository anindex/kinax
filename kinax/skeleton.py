import functools
from typing import Any, List, Optional, Sequence, Tuple, Union, NamedTuple

import matplotlib.pyplot as plt

from jax import numpy as jnp
from brax.base import Transform, Motion

from kinax.kinematics import forward
from kinax.model import URDFSystem


class Skeleton(NamedTuple):
    """Skeleton of a system."""
    # joint_names: List[str] = []
    joint_parents: List[int] = []
    joint_frame_pose: Transform = None
    joint_frame_vel: Motion = None


    def set(self, **kwargs: Any) -> "Skeleton":
        """Return a copy of self, possibly with overwrites."""
        return self._replace(**kwargs)

    def draw_skeleton(self, ax=None, c='k', **kwargs):
        """Draw skeleton."""
        if ax is None:
            ax = plt.gca()
        pos = self.joint_frame_pose.pos
        ax.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=10, c='b', **kwargs)
        # plot edges
        for i, parent in enumerate(self.joint_parents):
            if parent != -1:
                ax.plot3D([pos[i, 0], pos[parent, 0]],
                          [pos[i, 1], pos[parent, 1]],
                          [pos[i, 2], pos[parent, 2]],
                          linewidth=7, c=c, **kwargs)

    def __repr__(self) -> str:
        return f"Skeleton(joint_parents={self.joint_parents})"


def get_skeleton_from_system(sys: URDFSystem, q: jnp.array, qd: Optional[jnp.array] = None) -> Skeleton:
    """JITable get skeleton from system."""
    if qd is None:
        qd = jnp.zeros_like(q)
    x, xd = forward(sys, q, qd)
    return Skeleton(# joint_names=sys.link_names,
                    joint_parents=sys.link_parents,
                    joint_frame_pose=x,
                    joint_frame_vel=xd)
