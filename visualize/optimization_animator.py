"""Generate a GIF visualizing the optimization process.

For each optimization step, the GIF shows:
1. Initial particle layout (static)
2. Simulation replay (sampled from trajectory)
3. Transition to the next step (linear interpolation)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

from src.simulation import simulate


class OptimizationAnimator:
    """Build and save an optimization GIF from a position history.

    Parameters
    ----------
    pos_history : ndarray, shape [N_steps+1, num_ptl, 2]
        Particle positions at each optimization step (including final).
    vel_ptl, mass_ptl : jax arrays
        Fixed velocity and mass for all simulations.
    pos_bnd : jax array, shape [num_bnd, 2]
        Boundary positions.
    mass_bnd : jax array, shape [num_bnd,]
        Boundary masses.
    sim_params : dict
        Keys: h, g, dt, rho0, c0, gamma, n_steps, shepard_step.
    save_path : str
        Where to save the GIF.
    fps : int
        Frames per second (default 30).
    initial_duration : float
        Seconds to show the initial layout (default 0.3).
    sim_duration : float
        Seconds to play the simulation (default 0.5).
    transition_duration : float
        Seconds for the transition animation (default 0.3).
    x, y : float
        Dam dimensions for axis limits.
    ptl_s, bnd_s : float
        Scatter point sizes for particles / boundary.
    """

    def __init__(self, pos_history, vel_ptl, mass_ptl, pos_bnd, mass_bnd,
                 sim_params, save_path, *, fps=30,
                 initial_duration=0.3, sim_duration=0.5,
                 transition_duration=0.3, x=5.0, y=3.0,
                 ptl_s=0.1, bnd_s=0.1):
        self.pos_history = np.asarray(pos_history)
        self.vel_ptl = vel_ptl
        self.mass_ptl = mass_ptl
        self.pos_bnd = pos_bnd
        self.mass_bnd = mass_bnd
        self.sim_params = sim_params
        self.save_path = save_path
        self.fps = fps
        self.initial_duration = initial_duration
        self.sim_duration = sim_duration
        self.transition_duration = transition_duration
        self.x = x
        self.y = y
        self.ptl_s = ptl_s
        self.bnd_s = bnd_s

        self.n_initial = max(1, int(fps * initial_duration))
        self.n_sim = max(1, int(fps * sim_duration))
        self.n_transition = max(1, int(fps * transition_duration))

    # ------------------------------------------------------------------
    def _run_simulation(self, pos_ptl):
        """Run simulate() for one initial position and return trajectory."""
        import jax.numpy as jnp
        pos_ptl_jax = jnp.array(pos_ptl)
        trajectory = simulate(
            pos_ptl_jax, self.vel_ptl, self.mass_ptl,
            self.pos_bnd, self.mass_bnd,
            **self.sim_params)
        return np.asarray(trajectory)  # [n_steps, num_ptl, 5]

    # ------------------------------------------------------------------
    def _render_frame(self, positions, title):
        """Render particle positions to a PIL Image.

        Parameters
        ----------
        positions : ndarray, shape [num_ptl, 2]
        title : str
        """
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        scale = max(self.x, self.y)
        ax.set_xlim(-0.05 * scale, scale + 0.05 * scale)
        ax.set_ylim(-0.05 * scale, scale + 0.05 * scale)
        ax.set_aspect("equal")

        bnd_np = np.asarray(self.pos_bnd)
        ax.scatter(bnd_np[:, 0], bnd_np[:, 1], s=self.bnd_s, c="black")
        ax.scatter(positions[:, 0], positions[:, 1], s=self.ptl_s, c="blue")
        ax.set_title(title)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    # ------------------------------------------------------------------
    def create_gif(self):
        """Generate and save the full optimization GIF."""
        n_opt = len(self.pos_history)  # opt_steps + 1
        frames = []

        for i in range(n_opt):
            label = f"Step {i}/{n_opt - 1}"
            print(f"  GIF: processing {label} ...")

            pos = self.pos_history[i]  # [num_ptl, 2]

            # --- Phase 1: initial state (static) ---
            img = self._render_frame(pos, f"{label} - Initial")
            for _ in range(self.n_initial):
                frames.append(img)

            # --- Phase 2: simulation replay ---
            trajectory = self._run_simulation(pos)  # [n_steps, num_ptl, 5]
            total_sim_frames = trajectory.shape[0]
            indices = np.linspace(0, total_sim_frames - 1,
                                  self.n_sim, dtype=int)
            for idx in indices:
                sim_pos = trajectory[idx, :, :2]  # [num_ptl, 2]
                img = self._render_frame(sim_pos, f"{label} - Simulating")
                frames.append(img)

            # --- Phase 3: transition to next step ---
            if i < n_opt - 1:
                pos_next = self.pos_history[i + 1]
                for t in range(self.n_transition):
                    alpha = (t + 1) / self.n_transition
                    interp = (1 - alpha) * pos + alpha * pos_next
                    img = self._render_frame(interp, f"{label} - Transition")
                    frames.append(img)

        # --- Save GIF with Pillow ---
        duration_ms = 1000 // self.fps
        frames[0].save(
            self.save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0)
        print(f"  GIF saved to {self.save_path} "
              f"({len(frames)} frames, {self.fps} fps)")
