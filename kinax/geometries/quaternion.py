import functools
from typing import Any, List, Optional, Sequence, Tuple, Union


import jax
from jax import lax
from jax import numpy as jnp
from jax import vmap, jit
from brax import math


'''
"Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018. TABLE I
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
Format: q = [w, x, y, z]. # NOTE: change to [x, y, z, w] for pyBullet
'''


@jit
def sqrt_with_mask(x: jnp.ndarray) -> jnp.ndarray:
    ret = jnp.zeros_like(x)
    positive_mask = x > 0
    return ret.at[positive_mask].set(jnp.sqrt(x[positive_mask]))


@jit
def q_exp_map(v: jnp.ndarray, base: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    batch_dim = v.shape[:-1]
    if base is None:
        norm_v = jnp.linalg.norm(v, axis=-1)
        q = jnp.zeros(batch_dim + (4,))
        q = q.at[..., 0].set(1.)
        non_0 = jnp.where(norm_v > 0)
        non_0_norm = norm_v[non_0][..., None]
        q[non_0] = jnp.concatenate((
            jnp.cos(non_0_norm),
            (jnp.sin(non_0_norm) / non_0_norm) * v[non_0]), axis=-1)
        return q
    else:
        return q_mul(base, q_exp_map(v))


@jit
def q_log_map(q: jnp.ndarray, base: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    batch_dim = q.shape[:-1]
    if base is None:
        norm_q = jnp.linalg.norm(q, axis=-1)
        non_0 = jnp.where((norm_q > 0) * (jnp.abs(q[..., 0]) <= 1))  # eps for numerical stability
        q_non_singular = q[non_0]
        non_0_norm = norm_q[non_0][..., None]
        acos = jnp.arccos(q_non_singular[..., 0])[..., None]
        acos = acos.at[jnp.where(q_non_singular[..., 0] < 0)].add(-jnp.pi)  # q and -q maps to the same point in SO(3)
        v = jnp.zeros(batch_dim + (3,))
        # print(q_non_singular[..., 1:].shape, (acos / non_0_norm.unsqueeze(-1)).repeat((1,)*len(batch_dim) + (3,)).shape)
        v[non_0] = q_non_singular[..., 1:] * (acos / non_0_norm)
        return v
    else:
        return q_log_map(q_mul(q_inverse(base), q))

@jit
def q_parallel_transport(p_g: jnp.ndarray, g: jnp.ndarray, h: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    Q_g = q_to_quaternion_matrix(g)
    Q_h = q_to_quaternion_matrix(h)
    B = jnp.concatenate([
        jnp.zeros((1, 3)), jnp.eye(3)
    ], axis=0)
    log_g_h = q_log_map(h, base=g)
    m = jnp.linalg.norm(log_g_h, axis=-1)
    if m < eps:  # divide by zero
        return p_g
    q_temp = jnp.zeros((1, 4))
    q_temp = q_temp.at[0, 1:].set(log_g_h / m)
    u = Q_g @ q_temp
    I4 = jnp.eye(4)
    R_g_h = I4 - jnp.sin(m) * jnp.outer(g, u) + (jnp.cos(m) - 1) * jnp.outer(u, u)
    A_g_h = B.T @ Q_h.T @ R_g_h @ Q_g @ B
    res = (A_g_h @ p_g)
    return res


@jit
def q_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    res = q_to_quaternion_matrix(q1) @ q2
    return res


@jit
def q_inverse(q: jnp.ndarray) -> jnp.ndarray:
    scaling = jnp.tensor([1, -1, -1, -1])
    return q * scaling / q_norm_squared(q)


@jit
def q_div(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    return q_mul(q1, q_inverse(q2))


@jit
def q_norm_squared(q: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(q), axis=-1, keepdim=True)


def jax_unstack(x, axis=0):
  return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


@jit
def q_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = jax_unstack(q, axis=-1)
    double_cover = 2.0 / jnp.square(q).sum(-1)
    o = jnp.stack(
        (
            1 - double_cover * (y * y + z * z),
            double_cover * (x * y - z * w),
            double_cover * (x * z + y * w),
            double_cover * (x * y + z * w),
            1 - double_cover * (x * x + z * z),
            double_cover * (y * z - x * w),
            double_cover * (x * z - y * w),
            double_cover * (y * z + x * w),
            1 - double_cover * (x * x + y * y),
        ),
        axis=-1
    )
    return o.reshape(q.shape[:-1] + (3, 3))


@jit
def q_to_quaternion_matrix(q: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = jax_unstack(q, axis=-1)
    o = jnp.stack([
        w, -x, -y, -z,
        x, w, -z, y,
        y, z, w, -x,
        z, -y, x, w
    ], axis=-1)
    return o.reshape(q.shape[:-1] + (4, 4))


@jit
def q_to_axis_angles(q: jnp.ndarray, eps=1e-10) -> jnp.ndarray:
    norm_q = jnp.linalg.norm(q[..., 1:], axis=-1, keepdim=True)
    half_angles = jnp.arctan2(norm_q, q[..., :1])
    angles = 2 * half_angles
    beta = jnp.abs(angles) < eps
    s_half_angles = jnp.empty_like(angles)
    s_half_angles[~beta] = (
        jnp.sin(half_angles[~beta]) / angles[~beta]
    )
    s_half_angles[beta] = (
        0.5 - (angles[beta] * angles[beta]) / 48
    )
    return q[..., 1:] / s_half_angles


@jit
def axis_angles_to_q(axis_angles: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    angles = jnp.linalg.norm(axis_angles, axis=-1, keepdim=True)
    half_angles = angles / 2
    beta = jnp.abs(angles) < eps
    s_half_angles = jnp.empty_like(angles)
    s_half_angles[~beta] = (
        jnp.sin(half_angles[~beta]) / angles[~beta]
    )
    s_half_angles[beta] = (
        0.5 - (angles[beta] * angles[beta]) / 48
    )
    q = jnp.concatenate(
        [jnp.cos(half_angles), axis_angles * s_half_angles], axis=-1
    )
    return q


@jit
def q_to_euler(q: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = jax_unstack(q, axis=-1)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(t0, t1)
    t2 = jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = jnp.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(t3, t4)
    euler = jnp.stack([roll, pitch, yaw], axis=-1)
    return euler


@jit
def euler_to_q(euler: jnp.ndarray) -> jnp.ndarray:
    roll, pitch, yaw = jax_unstack(euler, axis=-1)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = jnp.stack([w, x, y, z], axis=-1)
    return q


@jit
def q_convert_xyzw(q: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = jax_unstack(q, axis=-1)
    return jnp.stack([x, y, z, w], axis=-1)


@jit
def q_convert_wxyz(q: jnp.ndarray) -> jnp.ndarray:
    x, y, z, w = jax_unstack(q, axis=-1)
    return jnp.stack([w, x, y, z], axis=-1)
