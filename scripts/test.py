import jax
from jax import numpy as jnp
from jax import vmap, jit

from kinax.geometries import distance

if __name__ == '__main__':
    # Test distance functions
    X = jnp.array([[0., 0., 0.], [1., 1., 1.]])
    Y = jnp.array([[1., 1., 1.], [0., 0., 0.]])
    print('euclidean_distance(X, Y) = ', distance.euclidean_distance(X, Y))
    print('euclidean_pairwise_distance(X, Y) = ', distance.euclidean_pairwise_distance(X, Y))
    
    # Test SE3 distance
    T1 = jnp.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    T2 = jnp.array([[1., 0., 0., 1.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 1.],
                    [0., 0., 0., 1.]])
    print(distance.SE3_distance(T1, T2))
