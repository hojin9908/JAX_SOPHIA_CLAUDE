"""Optimize initial particle positions via gradient descent on final max-y loss."""

import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from src.particle_gen import generate_boundary, generate_particles
from src.simulation import simulate_final

# jax.value_and_grad requires function whose argment is the variable to differentiate and only one argment.
# Therefore, loss_fn is wrapped by make_loss_fn to fix other arguments.
# Otherwise, it needs
#   def loss_fn(pos_ptl, vel_ptl, mass_ptl, pos_bnd, mass_bnd, h, g, dt, ...):
#   final_pos, _ = simulate_final(pos_ptl, vel_ptl, ...)
#   return jnp.max(final_pos[:, 1])
#   jax.value_and_grad(loss_fn, argnums=0)(pos_ptl, vel_ptl, mass_ptl, pos_bnd, ...)
def make_loss_fn(vel_ptl: jax.Array, mass_ptl: jax.Array,
                 pos_bnd: jax.Array, mass_bnd: jax.Array, *,
                 h: float, g: float, dt: float, rho0: float, c0: float,
                 gamma: float, n_steps: int, shepard_step: int,
                 checkpoint_every: int, use_soft_max: bool = False, temperature: float = 0.01) -> callable:
    """Build a loss function: pos_ptl -> scalar (max y of final water particles).
        '*" means the following arguments are passed as keyword arguments for better readability.
        for example, make_loss_fn(..., h=0.1, g=9.8, dt=1e-4, ...)

    Args:
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
        use_soft_max: If True, use logsumexp-based soft max for smoother gradients.
        temperature: Temperature for soft max (lower = closer to hard max).

    Returns:
        loss_fn(pos_ptl) -> scalar loss
    """
    def loss_fn(pos_ptl: jax.Array) -> jax.Array:
        """Compute loss (max y of final water particles) given initial particle positions.
        
        Args:
            pos_ptl: [num_ptl, 2] initial particle positions (variable for optimization)
        
        Returns:
            scalar loss (max y of final water particles)
        """
        final_pos, _ = simulate_final(
            pos_ptl, vel_ptl, mass_ptl, pos_bnd, mass_bnd,
            h=h, g=g, dt=dt, rho0=rho0, c0=c0, gamma=gamma,
            n_steps=n_steps, shepard_step=shepard_step,
            checkpoint_every=checkpoint_every)

        y_coords = final_pos[:, 1]

        # if use_soft_max:
        #     return temperature * logsumexp(y_coords / temperature)
        # else:
        #     return jnp.max(y_coords)
        return jnp.sum(y_coords)

    return loss_fn


def optimize(args):
    """Run optimization loop."""
    os.makedirs(args["save_dir"], exist_ok=True)

    # --- Particle generation ---
    spacing_approx = np.sqrt(args["x0"] * args["y0"] / args["num_ptl"])
    mass_approx = args["rho0"] * args["x0"] * args["y0"] / args["num_ptl"]

    pos_bnd, mass_bnd = generate_boundary(
        x=args["x"], y=args["y"],
        spacing=spacing_approx,
        bnd_layer=args["bnd_layer"],
        bnd_loss=args["bnd_loss"],
        mass=mass_approx)               # [num_bnd, 2], [num_bnd,]

    pos_ptl, vel_ptl, mass_ptl, actual_num = generate_particles(
        num=args["num_ptl"],
        x0=args["x0"], y0=args["y0"],
        rho0=args["rho0"],
        spacing=spacing_approx,
        uniform=args["uniform"],
        seed=args["seed"])              # [num_ptl, 2], [num_ptl, 2], [num_ptl,], int

    h = spacing_approx
    n_steps = int(args["t"] / args["dt"])

    # Ensure n_steps is divisible by checkpoint_every
    checkpoint_every = args["checkpoint_every"]
    if n_steps % checkpoint_every != 0:
        n_steps = (n_steps // checkpoint_every) * checkpoint_every
        print(f"Adjusted n_steps to {n_steps} (divisible by {checkpoint_every})")

    print(f"Smoothing length h = {h:.6f}")
    print(f"Number of time steps = {n_steps}")
    print(f"Particles: {actual_num}, Boundary: {pos_bnd.shape[0]}")
    print(f"Checkpoint every {checkpoint_every} steps "
          f"({n_steps // checkpoint_every} segments)")
    print(f"Learning rate: {args['lr']}, Optimization steps: {args['opt_steps']}")
    print(f"Soft max: {args['use_soft_max']}"
          + (f", temperature: {args['temperature']}" if args["use_soft_max"] else ""))

    # Save initial positions
    initial_pos_np = np.array(pos_ptl)      # [num_ptl, 2]
    np.save(os.path.join(args["save_dir"], "initial_pos.npy"), initial_pos_np)

    # --- Build loss and grad function ---
    loss_fn = make_loss_fn(
        vel_ptl, mass_ptl, pos_bnd, mass_bnd,
        h=h, g=args["g"], dt=args["dt"],
        rho0=args["rho0"], c0=args["c0"], gamma=args["gamma"],
        n_steps=n_steps, shepard_step=args["shepard_step"],
        checkpoint_every=checkpoint_every,
        use_soft_max=args["use_soft_max"],
        temperature=args["temperature"])


    lr = args["lr"]
    opt_steps = args["opt_steps"]

    @jax.jit
    def opt_step(pos_ptl):
        loss_val, grad_val = jax.value_and_grad(loss_fn)(pos_ptl)
        new_pos = pos_ptl - lr * grad_val
        grad_norm = jnp.linalg.norm(grad_val)
        return new_pos, loss_val, grad_norm

    # --- Optimization loop ---
    pos_history = []  # store positions at each optimization step

    print("\n--- Optimization ---")
    for step in range(opt_steps):
        pos_history.append(np.array(pos_ptl))

        t0 = time.time()
        pos_ptl, loss_val, grad_norm = opt_step(pos_ptl)
        # Block until computation completes for accurate timing
        loss_val.block_until_ready()
        t1 = time.time()

        print(f"Step {step:4d} | loss = {loss_val:.6f} | "
              f"|grad| = {grad_norm:.6e} | time = {t1 - t0:.2f}s")

    # Append final optimized positions
    pos_history.append(np.array(pos_ptl))

    # Save position history: [opt_steps+1, num_ptl, 2]
    pos_history_np = np.stack(pos_history, axis=0)
    np.save(os.path.join(args["save_dir"], "optimization_positions.npy"),
            pos_history_np)

    # --- Save optimized positions ---
    optimized_pos_np = np.array(pos_ptl)
    np.save(os.path.join(args["save_dir"], "optimized_pos.npy"), optimized_pos_np)
    print(f"\nSaved initial_pos.npy, optimized_pos.npy, and "
          f"optimization_positions.npy to {args['save_dir']}")

    # --- Optionally create optimization GIF ---
    if args["create_gif"]:
        from visualize.optimization_animator import OptimizationAnimator

        sim_params = dict(
            h=h, g=args["g"], dt=args["dt"],
            rho0=args["rho0"], c0=args["c0"], gamma=args["gamma"],
            n_steps=n_steps, shepard_step=args["shepard_step"])

        gif_path = os.path.join(args["save_dir"], "optimization.gif")
        print(f"\n--- Creating optimization GIF ---")
        animator = OptimizationAnimator(
            pos_history=pos_history_np,
            vel_ptl=vel_ptl,
            mass_ptl=mass_ptl,
            pos_bnd=pos_bnd,
            mass_bnd=mass_bnd,
            sim_params=sim_params,
            save_path=gif_path,
            x=args["x"],
            y=args["y"])
        animator.create_gif()


def parsing():
    parser = argparse.ArgumentParser(
        description="Optimize initial particle positions via gradient descent")

    # project property
    parser.add_argument("--tag", type=str, default="optimize")
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--seed", type=int, default=42)

    # simulation setting
    parser.add_argument("--x", type=float, default=5.0,
                        help="width of the dam")
    parser.add_argument("--y", type=float, default=3.0,
                        help="height of the dam")
    parser.add_argument("--x0", type=float, default=2.0,
                        help="width of initial particle distribution")
    parser.add_argument("--y0", type=float, default=1.0,
                        help="height of initial particle distribution")
    parser.add_argument("--uniform", action="store_true", default=True,
                        help="distribute initial particles uniformly")
    parser.add_argument("--shepard_step", type=int, default=10,
                        help="step interval for applying shepard filter")

    # physical coefficients
    parser.add_argument("--rho0", type=float, default=1000.0)
    parser.add_argument("--g", type=float, default=9.8)
    parser.add_argument("--c0", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=7.0)

    # particle configuration
    parser.add_argument("--num_ptl", type=int, default=1000)
    parser.add_argument("--bnd_layer", type=int, default=3)
    parser.add_argument("--bnd_loss", type=float, default=1.0)

    # PDE solver
    parser.add_argument("--t", type=float, default=0.5,
                        help="total simulation time")
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="time step")

    # optimization
    # Too small checkpoint_every may cause OOM during forward pass for saving states.
    # Too large checkpoint_every may cause OOM during backward pass for storing intermediates.
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="steps per checkpointed segment")
    parser.add_argument("--lr", type=float, default=1e-8,
                        help="learning rate")
    parser.add_argument("--opt_steps", type=int, default=10,
                        help="number of optimization iterations")
    parser.add_argument("--use_soft_max", action="store_true", default=False,
                        help="use logsumexp soft-max instead of hard max")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="temperature for soft-max")

    # visualization
    parser.add_argument("--create_gif", action="store_true", default=True,
                        help="create optimization GIF after optimization")

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parsing()
    optimize(args)
