import functools
from typing import Any, List, Optional, Sequence, Tuple, Union


import jax
from jax import numpy as jnp
from jax import vmap, jit
from brax import math


@jit
def euclidean_distance(X: jnp.ndarray, Y: jnp.ndarray, dim: int = -1) -> jnp.ndarray:
    """Compute euclidean distance between two sets of points.

    Args:
        X: first set of points
        Y: second set of points
        dim: dimension along which to compute distance

    Returns:
        euclidean distance between X and Y
    """
    return jnp.linalg.norm(X - Y, axis=dim)


@jit
def euclidean_pairwise_distance(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise euclidean distance between two sets of points.

    Args:
        X: first set of points
        Y: second set of points

    Returns:
        pairwise euclidean distance between X and Y
    """
    return jnp.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


@jit
def so3_relative_angle(R1: jnp.ndarray, R2: jnp.ndarray, eps: float = 1e-4):
    R12 = jnp.einsum('...ij,...jk->...ik', R1, R2)
    return so3_rotation_angle(R12, eps=eps)


@jit
def so3_rotation_angle(R: jnp.ndarray,  eps: float = 1e-4):
    '''
    Compute the rotation angle of a rotation matrix R.
    '''
    rot_trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # phi rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5
    return math.safe_arccos(jnp.clip(phi_cos, -1.0 + eps, 1.0 - eps))


@jit
def SE3_distance(T1: jnp.ndarray, T2: jnp.ndarray, w_pos: float = 1., w_rot: float = 1., eps: float = 1e-4):
    '''
    Compute the distance between two SE3 transforms.
    '''
    R1, R2 = T1[..., :3, :3], T2[..., :3, :3]
    p1, p2 = T1[..., :3, 3], T2[..., :3, 3]

    # rotation distance
    phi = so3_relative_angle(R1, R2, eps=eps)

    # translation distance
    p = p1 - p2
    p = jnp.linalg.norm(p, axis=-1)

    return w_pos * p + w_rot * phi
