"""
N-body simulation models.

GenericModel handles initial condition loading/generation, result saving,
and system resource & execution timing evaluation.
Subclasses implement the actual solvers for different processor targets.
"""

import os
import uuid
import time
import tracemalloc
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── Physics constants (defaults, can be overridden) ──────────────────────────
DEFAULT_MG = 0.1        # gravitational parameter (m*G product)
DEFAULT_R0 = 0.05       # softening / minimum interaction radius
DEFAULT_SIDE_LEN = 150  # domain side length for random initial conditions


class GenericModel(ABC):
    """
    Base class for all N-body solvers.

    Responsibilities
    ----------------
    * Load initial conditions from a .npy file  **or**  generate a random
      layout of *n_bodies* particles in a square domain.
    * Allocate position / velocity / acceleration arrays.
    * Provide the ``solve`` / ``evaluate`` / ``save`` interface.
    * Monitor execution time and system resources (CPU, RAM, throughput).
    * Persist results to ``results/<datetime>_<uuid>.npy``.

    Subclasses only need to implement ``_run_solver``.

    Parameters
    ----------
    positions : np.ndarray, shape (2, N)
        Initial (x, y) positions. Velocities always start at zero.
    dt : float
        Timestep.
    n_timesteps : int
        Number of integration steps.
    n_cores : int or None
        Number of CPU cores (relevant for parallel solvers).
    mG : float
        Gravitational parameter.
    r0 : float
        Softening radius.
    side_len : float
        Domain side length (used only when generating random ICs).
    """

    def __init__(
        self,
        positions: np.ndarray,
        dt: float,
        n_timesteps: int,
        *,
        n_cores: int | None = None,
        mG: float = DEFAULT_MG,
        r0: float = DEFAULT_R0,
        side_len: float = DEFAULT_SIDE_LEN,
    ):
        self.n_bodies = positions.shape[1]
        self.dt = dt
        self.n_timesteps = n_timesteps
        self.n_cores = n_cores
        self.mG = mG
        self.r0 = r0
        self.side_len = side_len

        # State arrays: shape (2, N, T+1)
        self.Pos = np.zeros((2, self.n_bodies, n_timesteps + 1))
        self.Vel = np.zeros((2, self.n_bodies, n_timesteps + 1))
        self.Pos[:, :, 0] = positions

        # Will be set after save()
        self._result_file: str | None = None

        # Evaluated performance statistics
        self.metrics: dict[str, float | int | str] | None = None

    # ── Factory helpers ──────────────────────────────────────────────────

    @staticmethod
    def load_initial_conditions(path: str) -> np.ndarray:
        """
        Load initial conditions from a ``.npy`` file.

        Expected shape: (N, 2)  →  returned as (2, N).
        """
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                f"Initial-condition file must have shape (N, 2), got {data.shape}"
            )
        return data.T  # → (2, N)

    @staticmethod
    def generate_random_positions(n_bodies: int, side_len: float = DEFAULT_SIDE_LEN) -> np.ndarray:
        """Return random (2, N) positions in [0, side_len)."""
        return side_len * np.random.rand(2, n_bodies)

    @classmethod
    def from_args(cls, args) -> "GenericModel":
        """
        Construct the right subclass from the parsed CLI ``args`` namespace.

        Dispatches on ``args.processor``.
        """
        # ── Initial conditions ───────────────────────────────────────────
        if args.initial_condition is not None:
            positions = cls.load_initial_conditions(args.initial_condition)
        else:
            positions = cls.generate_random_positions(args.n_bodies)

        # ── Common kwargs ────────────────────────────────────────────────
        kwargs = dict(
            positions=positions,
            dt=args.dt,
            n_timesteps=args.n_timesteps,
            n_cores=getattr(args, "n_cores", None),
        )

        # ── Dispatch to subclass ─────────────────────────────────────────
        processor_map: dict[str, type[GenericModel]] = {
            "cpu-serial": CPUSerialModel,
            "cpu-parallel": CPUParallelModel,
            "gpu": GPUModel,
        }
        subclass = processor_map.get(args.processor)
        if subclass is None:
            raise ValueError(f"Unknown processor: {args.processor}")
        return subclass(**kwargs)

    # ── Public interface ─────────────────────────────────────────────────

    def evaluate(self, *, save: bool = False, verbose: bool = True) -> dict[str, float | int | str]:
        """
        Execute the simulation while recording execution time and system resources.

        Parameters
        ----------
        save : bool
            Whether to persist result positions to disk.
        verbose : bool
            Whether to print a formatted summary report to stdout.

        Returns
        -------
        dict
            Dictionary containing timing, memory, and throughput metrics.
        """
        # Memory monitoring setup
        tracemalloc.start()
        proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        rss_start_mb = (proc.memory_info().rss / (1024 * 1024)) if proc else 0.0

        # Timing setup
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        # Run the solver
        self._run_solver()

        # Timing measurement
        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        wall_time = wall_end - wall_start
        cpu_time = cpu_end - cpu_start

        # Memory measurement
        _, peak_traced_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_heap_mb = peak_traced_bytes / (1024 * 1024)
        rss_end_mb = (proc.memory_info().rss / (1024 * 1024)) if proc else 0.0
        delta_rss_mb = max(0.0, rss_end_mb - rss_start_mb)

        # Derived metrics
        cpu_percent = (cpu_time / wall_time * 100.0) if wall_time > 0 else 0.0
        steps_per_sec = (self.n_timesteps / wall_time) if wall_time > 0 else 0.0
        time_per_step_ms = (wall_time / self.n_timesteps * 1000.0) if self.n_timesteps > 0 else 0.0

        # Pairwise force evaluations: N*(N-1)/2 per timestep
        total_pairs = (self.n_bodies * (self.n_bodies - 1) / 2.0) * self.n_timesteps
        interactions_per_sec = (total_pairs / wall_time) if wall_time > 0 else 0.0

        if save:
            self.save()

        self.metrics = {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "cpu_percent": cpu_percent,
            "peak_heap_mb": peak_heap_mb,
            "rss_mb": rss_end_mb,
            "delta_rss_mb": delta_rss_mb,
            "steps_per_sec": steps_per_sec,
            "time_per_step_ms": time_per_step_ms,
            "interactions_per_sec": interactions_per_sec,
            "n_bodies": self.n_bodies,
            "n_timesteps": self.n_timesteps,
            "dt": self.dt,
            "processor": self.__class__.__name__,
        }

        if verbose:
            print("\n" + "─" * 52)
            print(f" Simulation Performance  [{self.__class__.__name__}]")
            print("─" * 52)
            print(f" Particles (N)       : {self.n_bodies}")
            print(f" Timesteps (T)       : {self.n_timesteps} (dt = {self.dt})")
            print(f" Total wall time     : {wall_time:.4f} s")
            print(f" CPU time            : {cpu_time:.4f} s ({cpu_percent:.1f}% CPU)")
            print(f" Time per step       : {time_per_step_ms:.3f} ms")
            print(f" Simulation rate     : {steps_per_sec:.1f} steps/s")
            print(f" Pairwise throughput: {interactions_per_sec / 1e6:.2f} M pairs/s")
            print(f" Peak heap memory    : {peak_heap_mb:.2f} MB")
            if _HAS_PSUTIL:
                print(f" Process RSS memory  : {rss_end_mb:.2f} MB")
            print("─" * 52 + "\n")

        return self.metrics

    def solve(self, *, save: bool = False, verbose: bool = True) -> np.ndarray:
        """
        Run the simulation with timing/resource evaluation, optionally saving the result.

        Returns
        -------
        Pos : np.ndarray, shape (2, N, T+1)
        """
        self.evaluate(save=save, verbose=verbose)
        return self.Pos

    def save(self, directory: str = "results") -> str:
        """
        Persist ``self.Pos`` to ``<directory>/<datetime>_<uuid>.npy``.

        Returns the path to the saved file.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{timestamp}_{unique_id}.npy"
        path = os.path.join(directory, filename)
        np.save(path, self.Pos)
        self._result_file = path
        print(f"Results saved → {path}")
        return path

    @property
    def file(self) -> str | None:
        """Path to the last saved result file."""
        return self._result_file

    # ── To be implemented by subclasses ──────────────────────────────────

    @abstractmethod
    def _run_solver(self) -> None:
        """Run the N-body integration, filling ``self.Pos`` and ``self.Vel`` in-place."""
        ...


# ═════════════════════════════════════════════════════════════════════════════
#  Concrete solvers
# ═════════════════════════════════════════════════════════════════════════════

class CPUSerialModel(GenericModel):
    """
    Single-threaded CPU solver using Velocity-Verlet integration.

    This is a direct port of the original Main.py logic.
    """

    def _run_solver(self) -> None:
        Pos = self.Pos
        Vel = self.Vel
        dt = self.dt
        N = self.n_bodies
        mG = self.mG
        r0 = self.r0

        def calc_dist(current_pos):
            dist = current_pos.reshape(2, N, 1) - current_pos.reshape(2, 1, N)
            r = np.sqrt(dist[0] ** 2 + dist[1] ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_dist = dist / r
            return norm_dist, r

        def calc_acc(current_pos):
            norm_dist, r = calc_dist(current_pos)
            with np.errstate(divide="ignore", invalid="ignore"):
                acc = norm_dist * mG / (r ** 2)
            r_too_close = r < r0
            acc[:, r_too_close] = 0
            acc_summed = np.sum(acc, axis=1)
            return acc_summed

        def velocity_verlet(current_pos, current_vel, current_acc):
            predicted_pos = current_pos + current_vel * dt + 0.5 * current_acc * dt ** 2
            predicted_acc = calc_acc(predicted_pos)
            predicted_vel = current_vel + 0.5 * dt * (current_acc + predicted_acc)
            return predicted_pos, predicted_vel, predicted_acc

        current_acc = calc_acc(Pos[:, :, 0])
        for i in range(self.n_timesteps):
            Pos[:, :, i + 1], Vel[:, :, i + 1], current_acc = velocity_verlet(
                Pos[:, :, i], Vel[:, :, i], current_acc
            )


class CPUParallelModel(GenericModel):
    """
    Multi-core CPU solver (stub).

    TODO: implement parallel force computation (e.g. with multiprocessing,
          numba.prange, or joblib).
    """

    def _run_solver(self) -> None:
        raise NotImplementedError(
            "CPUParallelModel solver not yet implemented. "
            f"Would use {self.n_cores} cores."
        )


class GPUModel(GenericModel):
    """
    GPU-accelerated solver (stub).

    TODO: implement with CuPy, Numba CUDA, or PyCUDA.
    """

    def _run_solver(self) -> None:
        raise NotImplementedError("GPUModel solver not yet implemented.")
