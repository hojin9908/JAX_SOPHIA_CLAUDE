"""Particle generation for 2D dam-break SPH (NumPy init, returns JAX arrays)."""

import numpy as np
import jax
import jax.numpy as jnp


def generate_boundary(x: float, y: float, spacing: float, bnd_layer: int,
                      bnd_loss: float, mass: float) -> tuple[jax.Array, jax.Array]:
    """Generate 2D dam boundary particles.

    Creates left wall, bottom wall, and right wall with multiple layers.

    Args:
        x: dam width
        y: dam height
        spacing: base particle spacing
        bnd_layer: number of boundary layers
        bnd_loss: spacing multiplier for boundary particles
        mass: particle mass

    Returns:
        pos_bnd: [N_bnd, 2] boundary positions (JAX array)
        mass_bnd: [N_bnd] boundary masses (JAX array)
    """
    sp = bnd_loss * spacing

    # Left wall
    y1 = np.arange(0, y, sp)
    left_parts = []
    for i in range(bnd_layer):
        x_col = np.full_like(y1, -(i + 1) * sp)
        left_parts.append(np.stack([x_col, y1], axis=1))
    left = np.concatenate(left_parts, axis=0)

    # Bottom wall
    x2 = np.arange(-bnd_layer * sp, x + (bnd_layer + 1) * sp, sp)
    bottom_parts = []
    for i in range(bnd_layer):
        y_row = np.full_like(x2, -(i + 1) * sp)
        bottom_parts.append(np.stack([x2, y_row], axis=1))
    bottom = np.concatenate(bottom_parts, axis=0)

    # Right wall
    right_parts = []
    for i in range(bnd_layer):
        x_col = np.full_like(y1, x + (i + 1) * sp)
        right_parts.append(np.stack([x_col, y1], axis=1))
    right = np.concatenate(right_parts, axis=0)

    boundary = np.concatenate([left, bottom, right], axis=0)    # [N_bnd, 2]
    n_bnd = len(boundary)
    print(f"total number of boundary particle: {n_bnd:>6}")

    pos_bnd = jnp.array(boundary)       # [N_bnd, 2]
    mass_bnd = jnp.full(n_bnd, mass)    # [N_bnd]
    return pos_bnd, mass_bnd


def generate_particles(num: int, x0: float, y0: float, rho0: float,
                       spacing: float, uniform: bool = True,
                       seed: int = 42) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Generate initial fluid particles.

    Args:
        num: requested number of particles
        x0: width of initial particle distribution
        y0: height of initial particle distribution
        rho0: reference density
        spacing: particle spacing (used for uniform grid)
        uniform: if True, place particles on a regular grid
        seed: random seed for non-uniform placement

    Returns:
        pos: [N, 2] initial positions (JAX array)
        vel: [N, 2] initial velocities (zeros, JAX array)
        mass: [N] particle masses (JAX array)
        actual_num: actual number of particles created
    """
    if uniform:
        x_pts = np.arange(0, x0, spacing)
        y_pts = np.arange(0, y0, spacing)
        xx, yy = np.meshgrid(x_pts, y_pts)
        pts = np.c_[xx.ravel(), yy.ravel()]
        actual_num = len(pts)
        m = rho0 * x0 * y0 / actual_num
    else:
        np.random.seed(seed)
        pts = np.random.uniform(low=[0, 0], high=[x0, y0], size=(num, 2))
        actual_num = num
        m = rho0 * x0 * y0 / actual_num

    print(f"total number of simulation particle: {actual_num:>2}")

    pos = jnp.array(pts)                # [N, 2]
    vel = jnp.zeros_like(pos)           # [N, 2]
    mass_arr = jnp.full(actual_num, m)  # [N]
    return pos, vel, mass_arr, actual_num
