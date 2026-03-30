"""Benchmark recursive checkpointing: memory and time vs. checkpoint depth.

Each checkpoint depth is measured in an isolated subprocess so that JAX's
high-water-mark memory allocation doesn't carry over between runs.

Results are saved to result/checkpoint_benchmark.json and
result/checkpoint_benchmark.csv.

Usage:
    python benchmark_checkpoint.py [--num_ptl N] [--n_steps S]
                                    [--checkpoint_every C] [--max_depth D]
                                    [--save_dir PATH]

Internal (subprocess) usage:
    python benchmark_checkpoint.py --_single_depth D [other args...]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    import tracemalloc

from src.particle_gen import generate_boundary, generate_particles
from src.simulation import simulate_final_recursive


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _current_rss_mb():
    if _PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    return 0.0


class PeakMemoryTracker:
    """Thread-based peak RSS sampler (10 ms resolution)."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self._peak = _current_rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            rss = _current_rss_mb()
            if rss > self._peak:
                self._peak = rss
            self._stop.wait(self.interval)

    def __enter__(self):
        self._peak = _current_rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self):
        return self._peak


def _block(result):
    """Block until all JAX arrays in result are ready."""
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_and_measure(fn, *args):
    """Return (result, elapsed_s, peak_mb). Blocks on JAX computation."""
    if _PSUTIL:
        with PeakMemoryTracker() as tracker:
            t0 = time.perf_counter()
            result = fn(*args)
            _block(result)
            elapsed = time.perf_counter() - t0
        peak_mb = tracker.peak_mb
    else:
        tracemalloc.start()
        t0 = time.perf_counter()
        result = fn(*args)
        _block(result)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / 1024 ** 2
    return result, elapsed, peak_mb


# ---------------------------------------------------------------------------
# Single-depth worker (runs inside subprocess)
# ---------------------------------------------------------------------------

def _run_single_depth(args, depth):
    """Benchmark one checkpoint depth and print a JSON line to stdout."""
    spacing = np.sqrt(args["x0"] * args["y0"] / args["num_ptl"])
    mass_approx = args["rho0"] * args["x0"] * args["y0"] / args["num_ptl"]

    pos_bnd, mass_bnd = generate_boundary(
        x=args["x"], y=args["y"],
        spacing=spacing, bnd_layer=args["bnd_layer"],
        bnd_loss=1.0, mass=mass_approx)

    pos_ptl, vel_ptl, mass_ptl, actual_num = generate_particles(
        num=args["num_ptl"],
        x0=args["x0"], y0=args["y0"],
        rho0=args["rho0"], spacing=spacing,
        uniform=True, seed=42)

    h = spacing
    n_steps = args["n_steps"]
    checkpoint_every = args["checkpoint_every"]
    shepard_step = args["shepard_step"]

    sim_kw = dict(h=h, g=args["g"], dt=args["dt"],
                  rho0=args["rho0"], c0=args["c0"], gamma=args["gamma"],
                  n_steps=n_steps, shepard_step=shepard_step,
                  checkpoint_every=checkpoint_every, checkpoint_depth=depth)

    # ---- forward function ----
    @jax.jit
    def fwd_fn(pos):
        return simulate_final_recursive(pos, vel_ptl, mass_ptl, pos_bnd, mass_bnd,
                                        **sim_kw)

    # ---- value+grad function ----
    def loss(pos):
        final_pos, _ = simulate_final_recursive(pos, vel_ptl, mass_ptl,
                                                pos_bnd, mass_bnd, **sim_kw)
        return jnp.sum(final_pos[:, 1])

    @jax.jit
    def grad_fn(pos):
        return jax.value_and_grad(loss)(pos)

    # ---- warmup (triggers compilation) ----
    sys.stderr.write(f"[depth={depth}] compiling...\n"); sys.stderr.flush()
    _block(fwd_fn(pos_ptl))
    _block(grad_fn(pos_ptl))
    sys.stderr.write(f"[depth={depth}] compiled\n"); sys.stderr.flush()

    # ---- measure ----
    _, fwd_time, fwd_mem = run_and_measure(fwd_fn, pos_ptl)
    sys.stderr.write(f"[depth={depth}] fwd done\n"); sys.stderr.flush()
    _, grad_time, grad_mem = run_and_measure(grad_fn, pos_ptl)
    sys.stderr.write(f"[depth={depth}] grad done\n"); sys.stderr.flush()

    record = {
        "depth": depth,
        "checkpoint_every": checkpoint_every,
        "steps_per_top_segment": (checkpoint_every ** depth if depth > 0
                                   else n_steps),
        "n_steps": n_steps,
        "num_ptl": int(actual_num),
        "forward_time_s": round(fwd_time, 4),
        "forward_peak_mb": round(fwd_mem, 2),
        "grad_time_s": round(grad_time, 4),
        "grad_peak_mb": round(grad_mem, 2),
    }
    # Print to stdout so the parent process can parse it
    print(json.dumps(record))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main benchmark (spawns per-depth subprocesses)
# ---------------------------------------------------------------------------

def run_benchmark(args):
    os.makedirs(args["save_dir"], exist_ok=True)

    checkpoint_every = args["checkpoint_every"]
    n_steps = args["n_steps"]
    max_depth = args["max_depth"]

    # Determine valid depths
    depths = []
    for d in range(max_depth + 1):
        if d == 0:
            depths.append(d)
        else:
            divisor = checkpoint_every ** d
            if n_steps % divisor == 0:
                depths.append(d)
            else:
                print(f"Skipping depth={d}: n_steps={n_steps} not divisible by "
                      f"{checkpoint_every}^{d}={divisor}")

    print(f"n_steps={n_steps}, checkpoint_every={checkpoint_every}, "
          f"depths={depths}")
    print(f"Memory backend: {'psutil (RSS)' if _PSUTIL else 'tracemalloc'}")
    print("Each depth runs in an isolated subprocess for clean memory measurement.\n")

    records = []

    for depth in depths:
        print(f"--- depth={depth} ---", flush=True)

        # Build subprocess command that runs _run_single_depth
        cmd = [sys.executable, __file__,
               "--_single_depth", str(depth),
               "--num_ptl", str(args["num_ptl"]),
               "--n_steps", str(n_steps),
               "--checkpoint_every", str(checkpoint_every),
               "--max_depth", str(max_depth),
               "--shepard_step", str(args["shepard_step"]),
               "--x", str(args["x"]), "--y", str(args["y"]),
               "--x0", str(args["x0"]), "--y0", str(args["y0"]),
               "--rho0", str(args["rho0"]), "--g", str(args["g"]),
               "--c0", str(args["c0"]), "--gamma", str(args["gamma"]),
               "--dt", str(args["dt"]),
               "--bnd_layer", str(args["bnd_layer"]),
               "--save_dir", args["save_dir"]]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Print subprocess stderr (progress messages) to our stderr
        if proc.stderr:
            for line in proc.stderr.strip().splitlines():
                print(f"  {line}", flush=True)

        if proc.returncode != 0:
            print(f"  ERROR (returncode={proc.returncode})")
            print(f"  stdout: {proc.stdout[:500]}")
            print(f"  stderr: {proc.stderr[-500:]}")
            continue

        # Parse XLA rematerialization warning from stderr to get theoretical peak
        xla_original_gb = None
        xla_reduced_gb = None
        for line in proc.stderr.splitlines():
            if "rematerialization" in line and "originally" in line:
                import re
                m = re.search(r"reduced to\s+([\d.]+)GiB", line)
                m2 = re.search(r"down from\s+([\d.]+)GiB", line)
                if m:
                    xla_reduced_gb = float(m.group(1))
                if m2:
                    xla_original_gb = float(m2.group(1))

        # Parse JSON output from subprocess stdout
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    record = json.loads(line)
                    # Attach XLA memory estimates (None if below warning threshold)
                    record["xla_original_peak_gb"] = xla_original_gb
                    record["xla_reduced_peak_gb"] = xla_reduced_gb
                    records.append(record)
                    xla_str = (f"  XLA peak: {xla_original_gb:.2f} GB"
                               if xla_original_gb else "  XLA peak: <threshold")
                    print(f"  forward : {record['forward_time_s']:.3f}s  "
                          f"{record['forward_peak_mb']:.1f} MB (RSS)")
                    print(f"  fwd+bwd : {record['grad_time_s']:.3f}s  "
                          f"{record['grad_peak_mb']:.1f} MB (RSS)")
                    print(xla_str)
                except json.JSONDecodeError:
                    pass

    # ---- save results ----
    json_path = os.path.join(args["save_dir"], "checkpoint_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    csv_path = os.path.join(args["save_dir"], "checkpoint_benchmark.csv")
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    print(f"Saved CSV  → {csv_path}")

    # ---- summary table ----
    if records:
        print("\n--- Summary ---")
        print("(RSS = OS resident set size of subprocess; "
              "XLA peak = theoretical peak from XLA compiler analysis)")
        hdr = (f"{'depth':>6} {'steps/seg':>10} {'fwd_t(s)':>10} "
               f"{'grad_t(s)':>10} {'grad_RSS(MB)':>13} {'XLA_peak(GB)':>14}")
        print(hdr)
        print("-" * len(hdr))
        for r in records:
            xla = (f"{r['xla_original_peak_gb']:.2f}"
                   if r.get("xla_original_peak_gb") else "<thresh")
            print(f"{r['depth']:>6} {r['steps_per_top_segment']:>10} "
                  f"{r['forward_time_s']:>10.3f} "
                  f"{r['grad_time_s']:>10.3f} "
                  f"{r['grad_peak_mb']:>13.1f} "
                  f"{xla:>14}")
        print("\nNote: XLA peak is only reported when the compiler estimates "
              ">threshold memory;\n"
              "      lower depths reduce XLA's theoretical gradient memory footprint.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parsing():
    parser = argparse.ArgumentParser(
        description="Benchmark recursive checkpointing memory and time")
    # Internal flag: run single depth in subprocess mode
    parser.add_argument("--_single_depth", type=int, default=None,
                        help=argparse.SUPPRESS)

    parser.add_argument("--num_ptl", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--shepard_step", type=int, default=10)
    # domain
    parser.add_argument("--x", type=float, default=5.0)
    parser.add_argument("--y", type=float, default=3.0)
    parser.add_argument("--x0", type=float, default=2.0)
    parser.add_argument("--y0", type=float, default=1.0)
    # physics
    parser.add_argument("--rho0", type=float, default=1000.0)
    parser.add_argument("--g", type=float, default=9.8)
    parser.add_argument("--c0", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=7.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--bnd_layer", type=int, default=3)
    # output
    parser.add_argument("--save_dir", type=str, default="./result")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parsing()
    single_depth = args.pop("_single_depth")

    if single_depth is not None:
        # Subprocess mode: benchmark one depth and print JSON
        _run_single_depth(args, single_depth)
    else:
        run_benchmark(args)
