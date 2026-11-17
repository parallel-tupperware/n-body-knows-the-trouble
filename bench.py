#!/usr/bin/env python3
"""
N-body simulation scaling benchmark and complexity analysis tool.

Runs simulations across varying particle counts (N), analyzes scaling behavior,
fits empirical power-law complexity models against theoretical O(N^2), and
plots a comprehensive multi-panel performance visualization.

Usage examples
--------------
# Default benchmark across N = [50, 100, 200, 400, 600, 800, 1000]
python bench.py

# Custom particle counts and timesteps
python bench.py --processor cpu-serial --n-bodies 100 200 400 800 1200 --n-timesteps 100 --repeats 3

# Fast test run
python bench.py --n-bodies 50 100 200 400 --n-timesteps 50 --repeats 2
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from models import GenericModel


DEFAULT_N_BODIES = [50, 100, 200, 400, 600, 800, 1000]
DEFAULT_TIMESTEPS = 100
DEFAULT_DT = 0.05
DEFAULT_REPEATS = 3
DEFAULT_WARMUP = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark N-body simulation scaling and analyze computational complexity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processor",
        choices=["cpu-serial", "cpu-parallel", "gpu"],
        default="cpu-serial",
        help="Processor solver back-end to benchmark (default: cpu-serial).",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Number of CPU cores (for cpu-parallel).",
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        nargs="+",
        default=DEFAULT_N_BODIES,
        help=f"List of particle counts to test (default: {DEFAULT_N_BODIES}).",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Number of timesteps per run (default: {DEFAULT_TIMESTEPS}).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help=f"Integrator timestep (default: {DEFAULT_DT}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repetitions per N value for statistical averaging (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup runs per N value before recording times (default: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="benchmarks/complexity_scaling.png",
        help="Path where the generated plot will be saved (default: benchmarks/complexity_scaling.png).",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="benchmarks/scaling_data.csv",
        help="Path where CSV benchmark results will be saved (default: benchmarks/scaling_data.csv).",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    """Execute the benchmark suite across all requested particle counts."""
    results = []

    print("\n" + "=" * 70)
    print(f" Starting N-Body Scaling Benchmark  [{args.processor}]")
    print(f" Particle counts (N) : {args.n_bodies}")
    print(f" Timesteps (T)       : {args.n_timesteps} (dt = {args.dt})")
    print(f" Repetitions per N   : {args.repeats} (warmup = {args.warmup})")
    print("=" * 70 + "\n")

    header = f"{'N':>6} | {'Wall Time (s)':>15} | {'Std (s)':>10} | {'Time/Step (ms)':>16} | {'Peak Heap (MB)':>16} | {'Throughput (M/s)':>18}"
    print(header)
    print("─" * len(header))

    for N in args.n_bodies:
        # Run warmups
        for _ in range(args.warmup):
            model_args = argparse.Namespace(
                processor=args.processor,
                initial_condition=None,
                n_bodies=N,
                dt=args.dt,
                n_timesteps=args.n_timesteps,
                n_cores=args.n_cores,
            )
            model = GenericModel.from_args(model_args)
            model.evaluate(save=False, verbose=False)

        # Repetitions
        wall_times = []
        cpu_times = []
        step_times = []
        peak_mbs = []
        rss_mbs = []
        throughputs = []

        for _ in range(args.repeats):
            model_args = argparse.Namespace(
                processor=args.processor,
                initial_condition=None,
                n_bodies=N,
                dt=args.dt,
                n_timesteps=args.n_timesteps,
                n_cores=args.n_cores,
            )
            model = GenericModel.from_args(model_args)
            stats = model.evaluate(save=False, verbose=False)

            wall_times.append(stats["wall_time_s"])
            cpu_times.append(stats["cpu_time_s"])
            step_times.append(stats["time_per_step_ms"])
            peak_mbs.append(stats["peak_heap_mb"])
            rss_mbs.append(stats["rss_mb"])
            throughputs.append(stats["interactions_per_sec"] / 1e6)

        entry = {
            "n_bodies": N,
            "wall_mean_s": float(np.mean(wall_times)),
            "wall_std_s": float(np.std(wall_times)),
            "cpu_mean_s": float(np.mean(cpu_times)),
            "step_time_mean_ms": float(np.mean(step_times)),
            "step_time_std_ms": float(np.std(step_times)),
            "peak_heap_mb": float(np.mean(peak_mbs)),
            "rss_mb": float(np.mean(rss_mbs)),
            "throughput_mean_m_per_s": float(np.mean(throughputs)),
            "throughput_std_m_per_s": float(np.std(throughputs)),
        }
        results.append(entry)

        print(
            f"{entry['n_bodies']:6d} | "
            f"{entry['wall_mean_s']:15.4f} | "
            f"{entry['wall_std_s']:10.4f} | "
            f"{entry['step_time_mean_ms']:16.3f} | "
            f"{entry['peak_heap_mb']:16.2f} | "
            f"{entry['throughput_mean_m_per_s']:18.2f}"
        )

    print("─" * len(header) + "\n")
    return results


def analyze_complexity(results: list[dict]) -> dict:
    """
    Fit empirical power law: T(N) = a * N^b  =>  log(T) = log(a) + b * log(N).
    Compare against theoretical O(N^2) quadratic scaling.
    """
    N_arr = np.array([r["n_bodies"] for r in results], dtype=float)
    T_arr = np.array([r["wall_mean_s"] for r in results], dtype=float)

    log_N = np.log(N_arr)
    log_T = np.log(T_arr)

    # Linear regression in log-log space: log(T) = slope * log(N) + intercept
    coeffs = np.polyfit(log_N, log_T, 1)
    b_empirical = coeffs[0]
    log_a = coeffs[1]
    a_empirical = np.exp(log_a)

    # Predicted times from empirical fit
    T_fit_empirical = a_empirical * (N_arr ** b_empirical)

    # R^2 determination
    ss_tot = np.sum((T_arr - np.mean(T_arr)) ** 2)
    ss_res_empirical = np.sum((T_arr - T_fit_empirical) ** 2)
    r2_empirical = 1.0 - (ss_res_empirical / ss_tot) if ss_tot > 0 else 1.0

    # Theoretical O(N^2) fit: T = c * N^2  (linear regression through origin on N^2)
    c_theoretical = np.sum(T_arr * (N_arr ** 2)) / np.sum((N_arr ** 2) ** 2)
    T_fit_theoretical = c_theoretical * (N_arr ** 2)
    ss_res_theor = np.sum((T_arr - T_fit_theoretical) ** 2)
    r2_theoretical = 1.0 - (ss_res_theor / ss_tot) if ss_tot > 0 else 1.0

    print("=" * 70)
    print(" Computational Complexity Analysis")
    print("=" * 70)
    print(f" Empirical Model      : T(N) = {a_empirical:.2e} · N^{b_empirical:.2f}")
    print(f" Empirical Exponent   : b = {b_empirical:.2f}  (R² = {r2_empirical:.4f})")
    print(f" Theoretical Scaling  : O(N²) quadratic scaling (R² = {r2_theoretical:.4f})")
    if abs(b_empirical - 2.0) <= 0.15:
        print(" Verdict              : Closely matches theoretical O(N²) all-pairs scaling!")
    else:
        print(f" Note                 : Scaling exponent b={b_empirical:.2f} reflects cache/vectorization characteristics.")
    print("=" * 70 + "\n")

    return {
        "N_arr": N_arr,
        "T_arr": T_arr,
        "b_empirical": b_empirical,
        "a_empirical": a_empirical,
        "r2_empirical": r2_empirical,
        "c_theoretical": c_theoretical,
        "r2_theoretical": r2_theoretical,
    }


def save_csv_data(results: list[dict], filepath: str) -> None:
    """Save benchmark metrics to a CSV file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fields = [
        "n_bodies",
        "wall_mean_s",
        "wall_std_s",
        "cpu_mean_s",
        "step_time_mean_ms",
        "step_time_std_ms",
        "peak_heap_mb",
        "rss_mb",
        "throughput_mean_m_per_s",
        "throughput_std_m_per_s",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark CSV data saved → {filepath}")


def plot_benchmark_results(
    results: list[dict],
    analysis: dict,
    processor: str,
    n_timesteps: int,
    output_path: str,
) -> None:
    """Generate a multi-panel visualization of time complexity and system resources."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    N = analysis["N_arr"]
    T_mean = analysis["T_arr"]
    T_std = np.array([r["wall_std_s"] for r in results])
    b_emp = analysis["b_empirical"]
    a_emp = analysis["a_empirical"]
    r2_emp = analysis["r2_empirical"]
    c_theor = analysis["c_theoretical"]
    r2_theor = analysis["r2_theoretical"]

    peak_heap = np.array([r["peak_heap_mb"] for r in results])
    throughputs = np.array([r["throughput_mean_m_per_s"] for r in results])
    tp_std = np.array([r["throughput_std_m_per_s"] for r in results])

    # Dense array for smooth curve plotting
    N_dense = np.linspace(N.min(), N.max(), 200)
    T_dense_emp = a_emp * (N_dense ** b_emp)
    T_dense_theor = c_theor * (N_dense ** 2)

    # Style configuration
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"N-Body Simulation Complexity & Scaling Benchmark [{processor}]\n"
        f"T = {n_timesteps} steps, dt = {DEFAULT_DT}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # ── Panel 1: Execution Time vs N (Linear) ─────────────────────────────
    ax1 = axes[0, 0]
    ax1.errorbar(
        N, T_mean, yerr=T_std, fmt="o", color="#e74c3c", ecolor="#c0392b",
        capsize=5, label="Measured data (mean ± 1σ)", zorder=4
    )
    ax1.plot(
        N_dense, T_dense_emp, "-", color="#2980b9", lw=2.2,
        label=f"Empirical fit: $\\mathcal{{O}}(N^{{{b_emp:.2f}}})$ ($R^2={r2_emp:.4f}$)"
    )
    ax1.plot(
        N_dense, T_dense_theor, "--", color="#27ae60", lw=1.8,
        label=f"Theoretical: $\\mathcal{{O}}(N^2)$ ($R^2={r2_theor:.4f}$)"
    )
    ax1.set_title("Execution Time Scaling (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Bodies (N)", fontsize=11)
    ax1.set_ylabel("Total Wall Time (s)", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ── Panel 2: Log-Log Complexity Scaling ──────────────────────────────
    ax2 = axes[0, 1]
    ax2.errorbar(
        N, T_mean, yerr=T_std, fmt="s", color="#e74c3c", ecolor="#c0392b",
        capsize=5, label="Measured data", zorder=4
    )
    ax2.plot(
        N_dense, T_dense_emp, "-", color="#2980b9", lw=2.2,
        label=f"Fitted slope: $b={b_emp:.2f}$"
    )
    ax2.plot(
        N_dense, T_dense_theor, "--", color="#27ae60", lw=1.8,
        label="Theoretical slope: $b=2.00$"
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Complexity Scaling (Log-Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Bodies (N) [log]", fontsize=11)
    ax2.set_ylabel("Total Wall Time (s) [log]", fontsize=11)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    # ── Panel 3: Pairwise Interaction Throughput ─────────────────────────
    ax3 = axes[1, 0]
    ax3.errorbar(
        N, throughputs, yerr=tp_std, fmt="-^", color="#8e44ad", ecolor="#6c3483",
        capsize=5, lw=2, markersize=7, label="Pairwise Throughput"
    )
    ax3.set_title("Computational Throughput vs. Particle Count", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Number of Bodies (N)", fontsize=11)
    ax3.set_ylabel("Throughput ($10^6$ pairs / sec)", fontsize=11)
    ax3.legend(loc="lower right", framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # ── Panel 4: Memory Scaling ──────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.plot(N, peak_heap, "-d", color="#d35400", lw=2, markersize=7, label="Peak Heap Memory")
    ax4.set_title("Peak Heap Memory Usage vs. Particle Count", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Number of Bodies (N)", fontsize=11)
    ax4.set_ylabel("Peak Traced Heap (MB)", fontsize=11)
    ax4.legend(loc="upper left", framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Benchmark plot successfully saved → {output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 1. Run benchmarks
    results = run_benchmark(args)

    # 2. Analyze scaling
    analysis = analyze_complexity(results)

    # 3. Save raw CSV data
    if args.output_data:
        save_csv_data(results, args.output_data)

    # 4. Generate & save plot
    plot_benchmark_results(
        results=results,
        analysis=analysis,
        processor=args.processor,
        n_timesteps=args.n_timesteps,
        output_path=args.output_plot,
    )


if __name__ == "__main__":
    main()
