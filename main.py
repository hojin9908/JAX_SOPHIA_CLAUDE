"""Entry point for differentiable JAX SPH dam-break simulation."""

import argparse
import os
import time

import numpy as np
import jax

from src.particle_gen import generate_boundary, generate_particles
from src.simulation import simulate


def parsing():
    parser = argparse.ArgumentParser(description="Dam break SPH model (JAX)")

    # project property
    parser.add_argument("--tag", type=str, default="dam_break")
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
    parser.add_argument("--t", type=float, default=4.0,
                        help="total simulation time")
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="time step")

    # animation
    parser.add_argument("--animate", action="store_true", default=True,
                        help="create animation after simulation")
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--path", type=str, default="./result")
    parser.add_argument("--name", type=str, default="SPH_simulation")
    parser.add_argument("--ptl_s", type=float, default=0.1)
    parser.add_argument("--bnd_s", type=float, default=0.1)

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parsing()
    os.makedirs(args["save_dir"], exist_ok=True)

    # --- Particle generation ---
    spacing_approx = np.sqrt(args["x0"] * args["y0"] / args["num_ptl"])
    mass_approx = args["rho0"] * args["x0"] * args["y0"] / args["num_ptl"]

    pos_bnd, mass_bnd = generate_boundary(
        x=args["x"], y=args["y"],
        spacing=spacing_approx,
        bnd_layer=args["bnd_layer"],
        bnd_loss=args["bnd_loss"],
        mass=mass_approx)

    pos_ptl, vel_ptl, mass_ptl, actual_num = generate_particles(
        num=args["num_ptl"],
        x0=args["x0"], y0=args["y0"],
        rho0=args["rho0"],
        spacing=spacing_approx,
        uniform=args["uniform"],
        seed=args["seed"])

    # Smoothing length = particle spacing
    h = spacing_approx
    n_steps = int(args["t"] / args["dt"])

    print(f"Smoothing length h = {h:.6f}")
    print(f"Number of time steps = {n_steps}")
    print(f"Particles: {actual_num}, Boundary: {pos_bnd.shape[0]}")

    # --- Run simulation ---
    t0 = time.time()
    trajectory = simulate(
        pos_ptl, vel_ptl, mass_ptl, pos_bnd, mass_bnd,
        h=h, g=args["g"], dt=args["dt"],
        rho0=args["rho0"], c0=args["c0"], gamma=args["gamma"],
        n_steps=n_steps, shepard_step=args["shepard_step"])
    # Block until computation completes
    trajectory.block_until_ready()
    t1 = time.time()
    print(f"Simulation completed in {t1 - t0:.2f}s")
    print(f"Trajectory shape: {trajectory.shape}")

    # --- Save ---
    traj_np = np.array(trajectory)
    save_path = os.path.join(args["save_dir"], "simulation_trajectory.npy")
    np.save(save_path, traj_np)
    print(f"Saved trajectory to {save_path}")

    # --- Optional animation ---
    if args["animate"]:
        from visualize.animate import animate
        bnd_np = np.array(pos_bnd)
        # Pad boundary to 5 columns (x, y, 0, 0, mass) for compatibility
        bnd_5col = np.column_stack([
            bnd_np,
            np.zeros((len(bnd_np), 2)),
            np.array(mass_bnd)[:, None]])
        anim = animate(
            data=traj_np, bnd=bnd_5col,
            interval=args["interval"], fps=args["fps"],
            path=args["path"], name=args["name"],
            x=args["x"], y=args["y"],
            t=args["t"], dt=args["dt"],
            ptl_s=args["ptl_s"], bnd_s=args["bnd_s"])
        anim.animation_create()
        print("Animation saved.")
