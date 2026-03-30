"""Differentiable SPH simulation in pure JAX using lax.scan."""

import jax
import jax.numpy as jnp
from jax import lax

from .kernels import kernel_matrix, grad_kernel_matrix
from .eos import tait_eos


def _compute_shepard(pos_ptl: jax.Array, pos_bnd: jax.Array,
                     mass_ptl: jax.Array, mass_bnd: jax.Array,
                     rho_ptl: jax.Array, rho_bnd: jax.Array,
                     h: float) -> tuple[jax.Array, jax.Array]:
    """Compute Shepard correction filters for particles and boundary.

    filter_i = sum_j (m_j / rho_j) * W_ij

    Returns:
        ptl_filter: [num_ptl]
        bnd_filter: [num_bnd]
    """
    # Kernel matrices
    W_pp = kernel_matrix(pos_ptl, pos_ptl, h)   # [num_ptl, num_ptl]
    W_pb = kernel_matrix(pos_ptl, pos_bnd, h)   # [num_ptl, num_bnd]
    W_bp = kernel_matrix(pos_bnd, pos_ptl, h)   # [num_bnd, num_ptl]
    W_bb = kernel_matrix(pos_bnd, pos_bnd, h)   # [num_bnd, num_bnd]

    # m/rho ratios
    mr_ptl = mass_ptl / rho_ptl   # [num_ptl]
    mr_bnd = mass_bnd / rho_bnd   # [num_bnd]

    # Shepard filter for particles: sum over ptl neighbors + bnd neighbors
    ptl_filter = (W_pp * mr_ptl[None, :]).sum(axis=1) + \
                 (W_pb * mr_bnd[None, :]).sum(axis=1)  # [num_ptl]

    # Shepard filter for boundary
    bnd_filter = (W_bp * mr_ptl[None, :]).sum(axis=1) + \
                 (W_bb * mr_bnd[None, :]).sum(axis=1)  # [num_bnd]

    return ptl_filter, bnd_filter   # [num_ptl], [num_bnd]


def _make_step_fn(pos_bnd, mass_ptl, mass_bnd, *,
                  h, g, dt, rho0, c0, gamma, shepard_step,
                  accumulate_state=True):
    """Factory that builds a single SPH time-step function.

    Args:
        pos_bnd: [num_bnd, 2] boundary positions (fixed)
        mass_ptl: [num_ptl] particle masses
        mass_bnd: [num_bnd] boundary masses
        h: smoothing length
        g: gravitational acceleration
        dt: time step
        rho0: reference density
        c0: speed of sound
        gamma: EOS stiffness parameter
        shepard_step: recompute Shepard filter every this many steps
        accumulate_state: If True, return [num_ptl, 5] state as scan output
                          (for trajectory recording). If False, return None
                          (for optimization — avoids O(n_steps) memory).

    Returns:
        step_fn(carry, step_idx) compatible with lax.scan.
    """
    grav = jnp.array([0.0, -g])

    def step_fn(carry, step_idx):
        """
        Simulate one time step of SPH.
        Args:
            carry: tuple containing current state:
                pos_p: [num_ptl, 2] particle positions
                vel_p: [num_ptl, 2] particle velocities
                rho_p: [num_ptl] particle densities
                rho_b: [num_bnd] boundary densities
                s_ptl: [num_ptl] particle Shepard filter
                s_bnd: [num_bnd] boundary Shepard filter
            step_idx: scalar index of current step (for conditional Shepard update)
        Returns:
            new_carry: updated state tuple
            state_out: if accumulate_state, [num_ptl, 5] (x, y, vx, vy, mass); else None

        """
        pos_p, vel_p, rho_p, rho_b, s_ptl, s_bnd = carry    # [num_ptl, 2], [num_ptl, 2], [num_ptl], [num_bnd], [num_ptl], [num_bnd]

        # --- Conditionally update Shepard filter ---
        def update_shepard(_):
            return _compute_shepard(
                pos_p, pos_bnd, mass_ptl, mass_bnd, rho_p, rho_b, h)

        def keep_shepard(_):
            return s_ptl, s_bnd

        s_ptl, s_bnd = lax.cond(
            step_idx % shepard_step == 0,
            update_shepard,
            keep_shepard,
            operand=None)

        # --- Kernel matrices for density ---
        W_pp = kernel_matrix(pos_p, pos_p, h)     # [num_ptl, num_ptl]
        W_pb = kernel_matrix(pos_p, pos_bnd, h)   # [num_ptl, num_bnd]
        W_bp = kernel_matrix(pos_bnd, pos_p, h)    # [num_bnd, num_ptl]
        W_bb = kernel_matrix(pos_bnd, pos_bnd, h)  # [num_bnd, num_bnd]

        # Normalized kernel (Shepard corrected)
        W_pp_n = W_pp / s_ptl[:, None]  # [num_ptl, num_ptl]
        W_pb_n = W_pb / s_ptl[:, None]  # [num_ptl, num_bnd]
        W_bp_n = W_bp / s_bnd[:, None]  # [num_bnd, num_ptl]
        W_bb_n = W_bb / s_bnd[:, None]  # [num_bnd, num_bnd]

        # --- Density ---
        rho_p_new = (W_pp_n * mass_ptl[None, :]).sum(axis=1) + \
                    (W_pb_n * mass_bnd[None, :]).sum(axis=1)  # [num_ptl]
        rho_b_new = (W_bp_n * mass_ptl[None, :]).sum(axis=1) + \
                    (W_bb_n * mass_bnd[None, :]).sum(axis=1)  # [num_bnd]

        # --- Pressure (Tait EOS) ---
        pres_p = tait_eos(rho_p_new, rho0, c0, gamma)  # [num_ptl]
        pres_b = tait_eos(rho_b_new, rho0, c0, gamma)  # [num_bnd]

        # --- Gradient kernel matrices for pressure force ---
        gW_pp = grad_kernel_matrix(pos_p, pos_p, h)     # [num_ptl, num_ptl, 2]
        gW_pb = grad_kernel_matrix(pos_p, pos_bnd, h)   # [num_ptl, num_bnd, 2]

        # Shepard-correct the gradient kernels
        gW_pp_n = gW_pp / s_ptl[:, None, None]  # [num_ptl, num_ptl, 2]
        gW_pb_n = gW_pb / s_ptl[:, None, None]  # [num_ptl, num_bnd, 2]

        # --- Pressure acceleration ---
        coeff_pp = mass_ptl[None, :] * (pres_p[:, None] + pres_p[None, :]) / rho_p_new[None, :]  # [num_ptl, num_ptl]
        accel_pp = jnp.einsum('ij,ijd->id', coeff_pp, gW_pp_n)  # [num_ptl, 2]

        coeff_pb = mass_bnd[None, :] * (pres_p[:, None] + pres_b[None, :]) / rho_b_new[None, :]  # [num_ptl, num_bnd]
        accel_pb = jnp.einsum('ij,ijd->id', coeff_pb, gW_pb_n)  # [num_ptl, 2]

        accel_pres = -(accel_pp + accel_pb) / rho_p_new[:, None]  # [num_ptl, 2]

        # --- Total acceleration ---
        accel = accel_pres + grav[None, :]

        # --- Symplectic Euler integration ---
        vel_new = vel_p + accel * dt
        pos_new = pos_p + vel_new * dt

        new_carry = (pos_new, vel_new, rho_p_new, rho_b_new, s_ptl, s_bnd)  # [num_ptl, 2], [num_ptl, 2], [num_ptl], [num_bnd], [num_ptl], [num_bnd]

        if accumulate_state:
            state = jnp.concatenate([pos_p, vel_p, mass_ptl[:, None]], axis=1)  # [num_ptl, 5]
            return new_carry, state
        else:
            return new_carry, None

    return step_fn


def simulate(pos_ptl: jax.Array, vel_ptl: jax.Array, mass_ptl: jax.Array,
             pos_bnd: jax.Array, mass_bnd: jax.Array, *,
             h: float, g: float, dt: float, rho0: float, c0: float,
             gamma: float, n_steps: int, shepard_step: int) -> jax.Array:
    """Run a differentiable SPH dam-break simulation.

    All state is passed functionally; no mutation. Uses lax.scan for
    reverse-mode AD compatibility through the full time loop.

    Args:
        pos_ptl: [N, 2] initial particle positions
        vel_ptl: [N, 2] initial particle velocities
        mass_ptl: [N] particle masses
        pos_bnd: [M, 2] boundary positions (fixed)
        mass_bnd: [M] boundary masses
        h: smoothing length
        g: gravitational acceleration
        dt: time step
        rho0: reference density
        c0: speed of sound
        gamma: EOS stiffness parameter
        n_steps: number of simulation steps
        shepard_step: recompute Shepard filter every this many steps

    Returns:
        trajectory: [n_steps, N, 5] — (x, y, vx, vy, mass) per step
    """
    n_ptl = pos_ptl.shape[0]

    # Initialize densities to rho0
    rho_ptl = jnp.full(n_ptl, rho0)
    rho_bnd = jnp.full(pos_bnd.shape[0], rho0)

    # Initial Shepard filter
    ptl_filter, bnd_filter = _compute_shepard(
        pos_ptl, pos_bnd, mass_ptl, mass_bnd, rho_ptl, rho_bnd, h)

    init_carry = (pos_ptl, vel_ptl, rho_ptl, rho_bnd, ptl_filter, bnd_filter)

    step_fn = _make_step_fn(pos_bnd, mass_ptl, mass_bnd,
                            h=h, g=g, dt=dt, rho0=rho0, c0=c0,
                            gamma=gamma, shepard_step=shepard_step,
                            accumulate_state=True)

    _, trajectory = lax.scan(step_fn, init_carry, jnp.arange(n_steps))
    return trajectory  # [n_steps, N, 5]


def simulate_final(pos_ptl: jax.Array, vel_ptl: jax.Array,
                   mass_ptl: jax.Array, pos_bnd: jax.Array,
                   mass_bnd: jax.Array, *,
                   h: float, g: float, dt: float, rho0: float, c0: float,
                   gamma: float, n_steps: int, shepard_step: int,
                   checkpoint_every: int = 100) -> tuple[jax.Array, jax.Array]:
    """Run simulation and return only the final state (memory-efficient).

    Uses segmented gradient checkpointing: an outer lax.scan iterates over
    segments, each containing an inner lax.scan of `checkpoint_every` steps.
    jax.checkpoint is applied per segment so that backward-pass memory is
    O(n_segments + segment_size) instead of O(n_steps).

    Args:
        pos_ptl: [num_ptl, 2] initial particle positions
        vel_ptl: [num_ptl, 2] initial particle velocities
        mass_ptl: [num_ptl] particle masses
        pos_bnd: [num_bnd, 2] boundary positions (fixed)
        mass_bnd: [num_bnd] boundary masses
        h: smoothing length
        g: gravitational acceleration
        dt: time step
        rho0: reference density
        c0: speed of sound
        gamma: EOS stiffness parameter
        n_steps: number of simulation steps (must be divisible by checkpoint_every)
        shepard_step: recompute Shepard filter every this many steps
        checkpoint_every: number of steps per checkpointed segment

    Returns:
        final_pos: [num_ptl, 2] final particle positions
        final_vel: [num_ptl, 2] final particle velocities
    """
    assert n_steps % checkpoint_every == 0, (
        f"n_steps ({n_steps}) must be divisible by checkpoint_every ({checkpoint_every})")

    n_segments = n_steps // checkpoint_every
    n_ptl = pos_ptl.shape[0]

    # Initialize densities to rho0
    rho_ptl = jnp.full(n_ptl, rho0)                 # [num_ptl]
    rho_bnd = jnp.full(pos_bnd.shape[0], rho0)      # [num_bnd]

    # Initial Shepard filter
    ptl_filter, bnd_filter = _compute_shepard(
        pos_ptl, pos_bnd, mass_ptl, mass_bnd, rho_ptl, rho_bnd, h)      # [num_ptl], [num_bnd]

    init_carry = (pos_ptl, vel_ptl, rho_ptl, rho_bnd, ptl_filter, bnd_filter)

    step_fn = _make_step_fn(pos_bnd, mass_ptl, mass_bnd,
                            h=h, g=g, dt=dt, rho0=rho0, c0=c0,
                            gamma=gamma, shepard_step=shepard_step,
                            accumulate_state=False)

    @jax.checkpoint
    def segment_fn(carry, segment_idx):
        """Run one segment of checkpoint_every steps.
        for example, if checkpoint_every=100, 
                        segment_idx=0 runs steps [0..99],  
                        segment_idx=1 runs steps [100..199],   
                        etc.
        """
        local_indices = segment_idx * checkpoint_every + jnp.arange(checkpoint_every)   # Local step indices

        def inner_step(c, idx):
            new_c, _ = step_fn(c, idx)
            return new_c, None

        carry, _ = lax.scan(inner_step, carry, local_indices)
        return carry, None

    final_carry, _ = lax.scan(segment_fn, init_carry, jnp.arange(n_segments))
    final_pos, final_vel = final_carry[0], final_carry[1]
    return final_pos, final_vel


def _make_recursive_segment_fn(step_fn, checkpoint_every: int, depth: int):
    """Build a recursively checkpointed segment function.

    Returns fn(carry, start_idx) -> (carry, None) that processes
    checkpoint_every^depth steps starting at global step index start_idx.

    Args:
        step_fn: single-step function (carry, step_idx) -> (carry, None)
        checkpoint_every: branching factor at each level
        depth: number of checkpointing levels (>= 1)

    Levels:
        depth=1  — runs checkpoint_every leaf steps with @jax.checkpoint
        depth=k  — splits into checkpoint_every sub-segments of depth k-1,
                   wrapping each sub-call with @jax.checkpoint
    """
    if depth == 1:
        @jax.checkpoint
        def fn(carry, start_idx):
            local_indices = start_idx + jnp.arange(checkpoint_every)
            def inner(c, idx):
                new_c, _ = step_fn(c, idx)
                return new_c, None
            carry, _ = lax.scan(inner, carry, local_indices)
            return carry, None
        return fn
    else:
        steps_per_sub = int(checkpoint_every ** (depth - 1))
        sub_fn = _make_recursive_segment_fn(step_fn, checkpoint_every, depth - 1)

        @jax.checkpoint
        def fn(carry, start_idx):
            sub_starts = start_idx + jnp.arange(checkpoint_every) * steps_per_sub
            carry, _ = lax.scan(sub_fn, carry, sub_starts)
            return carry, None
        return fn


def simulate_final_recursive(
        pos_ptl: jax.Array, vel_ptl: jax.Array,
        mass_ptl: jax.Array, pos_bnd: jax.Array,
        mass_bnd: jax.Array, *,
        h: float, g: float, dt: float, rho0: float, c0: float,
        gamma: float, n_steps: int, shepard_step: int,
        checkpoint_every: int = 10,
        checkpoint_depth: int = 1) -> tuple[jax.Array, jax.Array]:
    """Run simulation with configurable recursive gradient checkpointing.

    Memory vs. compute trade-off:
        depth=0  — no checkpointing; O(n_steps) backward memory, 1x compute
        depth=1  — equivalent to simulate_final; O(n_steps/c) memory, ~2x compute
        depth=k  — k-level nesting; O(k * n_steps^(1/(k+1))) memory, ~(k+1)x compute

    where c = checkpoint_every.

    n_steps must be divisible by checkpoint_every^checkpoint_depth (for depth>0).

    Args:
        pos_ptl: [num_ptl, 2] initial particle positions
        vel_ptl: [num_ptl, 2] initial particle velocities
        mass_ptl: [num_ptl] particle masses
        pos_bnd: [num_bnd, 2] boundary positions (fixed)
        mass_bnd: [num_bnd] boundary masses
        h: smoothing length
        g: gravitational acceleration
        dt: time step
        rho0: reference density
        c0: speed of sound
        gamma: EOS stiffness parameter
        n_steps: total simulation steps
        shepard_step: recompute Shepard filter every this many steps
        checkpoint_every: branching factor at each checkpointing level
        checkpoint_depth: number of recursive checkpointing levels (0 = none)

    Returns:
        final_pos: [num_ptl, 2] final particle positions
        final_vel: [num_ptl, 2] final particle velocities
    """
    n_ptl = pos_ptl.shape[0]
    rho_ptl = jnp.full(n_ptl, rho0)
    rho_bnd = jnp.full(pos_bnd.shape[0], rho0)

    ptl_filter, bnd_filter = _compute_shepard(
        pos_ptl, pos_bnd, mass_ptl, mass_bnd, rho_ptl, rho_bnd, h)

    init_carry = (pos_ptl, vel_ptl, rho_ptl, rho_bnd, ptl_filter, bnd_filter)

    step_fn = _make_step_fn(pos_bnd, mass_ptl, mass_bnd,
                            h=h, g=g, dt=dt, rho0=rho0, c0=c0,
                            gamma=gamma, shepard_step=shepard_step,
                            accumulate_state=False)

    if checkpoint_depth == 0:
        # No checkpointing: plain scan over all steps
        def plain_step(carry, idx):
            new_c, _ = step_fn(carry, idx)
            return new_c, None
        final_carry, _ = lax.scan(plain_step, init_carry, jnp.arange(n_steps))
    else:
        steps_per_top = int(checkpoint_every ** checkpoint_depth)
        assert n_steps % steps_per_top == 0, (
            f"n_steps ({n_steps}) must be divisible by "
            f"checkpoint_every^checkpoint_depth = {checkpoint_every}^{checkpoint_depth} = {steps_per_top}")
        n_top = n_steps // steps_per_top

        seg_fn = _make_recursive_segment_fn(step_fn, checkpoint_every, checkpoint_depth)

        # Pass global start indices directly so step_fn receives correct step numbers
        top_starts = jnp.arange(n_top) * steps_per_top

        def outer_step(carry, start_idx):
            carry, _ = seg_fn(carry, start_idx)
            return carry, None

        final_carry, _ = lax.scan(outer_step, init_carry, top_starts)

    final_pos, final_vel = final_carry[0], final_carry[1]
    return final_pos, final_vel
