"""Pure JAX Wendland C2 kernel and its gradient for 2D SPH."""

import jax
import jax.numpy as jnp

EPS = 1e-6


def safe_norm(dx: jax.Array, eps: float = EPS) -> jax.Array:
    """Differentiable L2 norm: sqrt(sum(x^2) + eps^2) to avoid NaN grad at r=0."""
    return jnp.sqrt(jnp.sum(dx ** 2, axis=-1) + eps ** 2)


def wendland_c2(r: jax.Array, h: float) -> jax.Array:
    """Wendland C2 kernel (2D).

    W(r, h) = (7 / (4*pi*h^2)) * (1 - r/(2h))^4 * (1 + 4*r/(2h))
    Compact support: r < 2h.
    """
    q = r / (2.0 * h)
    const = 7.0 / (jnp.pi * 4.0 * h ** 2)
    value = const * (1.0 - q) ** 4 * (1.0 + 4.0 * q)
    return jnp.where(r < 2.0 * h, value, 0.0)


def kernel_matrix(pos_i: jax.Array, pos_j: jax.Array, h: float) -> jax.Array:
    """All-pairs kernel evaluation.

    Args:
        pos_i: [N, 2] positions of subject particles
        pos_j: [M, 2] positions of neighbor particles
        h: smoothing length

    Returns:
        W: [N, M] kernel values
    """
    dx = pos_i[:, None, :] - pos_j[None, :, :]   # [N, M, 2]
    r = safe_norm(dx)                              # [N, M]
    return wendland_c2(r, h)


def grad_kernel_matrix(pos_i: jax.Array, pos_j: jax.Array, h: float) -> jax.Array:
    """All-pairs gradient of Wendland C2 kernel w.r.t. pos_i.

    grad_i W = dW/dr * (x_i - x_j) / r

    For Wendland C2:
        dW/dr = const/(2h) * (-20) * (1 - q)^3 * q   where q = r/(2h)

    Args:
        pos_i: [N, 2]
        pos_j: [M, 2]
        h: smoothing length

    Returns:
        gradW: [N, M, 2]
    """
    dx = pos_i[:, None, :] - pos_j[None, :, :]   # [N, M, 2]
    r = safe_norm(dx)                              # [N, M]
    q = r / (2.0 * h)
    const = 7.0 / (jnp.pi * 4.0 * h ** 2)
    # dW/dr = const/(2h) * (-20) * (1 - q)^3 * q
    dWdr = const / (2.0 * h) * (-20.0) * (1.0 - q) ** 3 * q  # [N, M]
    # grad_i W = dW/dr * dx / r
    grad = dWdr[..., None] * dx / r[..., None]     # [N, M, 2]
    return jnp.where((r < 2.0 * h)[..., None], grad, 0.0)   # [N, M, 2]
