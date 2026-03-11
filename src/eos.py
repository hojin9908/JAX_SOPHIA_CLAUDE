"""Equation of State functions for SPH pressure calculation."""

import jax
import jax.numpy as jnp


def tait_eos(rho: jax.Array, rho0: float, c0: float, gamma: float) -> jax.Array:
    """Tait equation of state.

    p = (c0^2 * rho0 / gamma) * ((rho / rho0)^gamma - 1)
    """
    const = c0 ** 2 * rho0 / gamma
    return const * ((rho / rho0) ** gamma - 1.0)


def tait_murnaghan_eos(rho: jax.Array, rho0: float, K0: float, gamma: float, p0: float) -> jax.Array:
    """Tait-Murnaghan equation of state.

    p = (K0 / gamma) * ((rho / rho0)^gamma - 1) + p0
    """
    const = K0 / gamma
    return const * ((rho / rho0) ** gamma - 1.0) + p0
