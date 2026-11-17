# N-Body Simulation Framework (`n-body-knows-the-trouble`)

High-performance 2D N-body gravitational simulation framework with modular solvers, dual-extent video rendering, and computational complexity benchmarking.

---

## 1. Environment Setup

The repository includes an `environment.yml` specifying all required dependencies, including `ffmpeg` for video encoding.

### Create & activate the conda environment:
```bash
# Clone the repository (if not already local) and cd into it:
cd ~/wi4049tu/n-body-knows-the-trouble

# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate nbody
```

---

## 2. Project Architecture

```
n-body-knows-the-trouble/
├── main.py              # Main simulation CLI entry point
├── models.py            # GenericModel ABC + CPUSerialModel / CPUParallelModel / GPUModel
├── renderers.py         # GenericRenderer ABC + DiscreteRenderer / DensityRenderer (+ CLI)
├── bench.py             # Benchmarking & complexity scaling analysis tool
├── environment.yml      # Conda environment definition
├── results/             # Saved simulation trajectories (.npy)
├── renders/             # Generated video animations (.mp4)
└── benchmarks/          # Benchmark scaling plots (.png) and metrics (.csv)
```

### Data Structures & File Conventions

- **Initial Conditions file** (`--initial-condition <path.npy>`):
  - 2D array of shape `(N, 2)`: each row is `[x, y]`.
  - Particles start with **zero initial velocity**.
  - If omitted, random coordinates are generated across a $[0, 150) \times [0, 150)$ domain.
- **Simulation Results** (`results/<timestamp>_<uuid>.npy`):
  - 3D array of shape `(2, N, T+1)`:
    - `data[0, :, t]`: x-coordinates of all $N$ particles at timestep $t$.
    - `data[1, :, t]`: y-coordinates of all $N$ particles at timestep $t$.
    - $t = 0$ is the initial condition; $t = T$ is the final step.

---

## 3. How to Use the Main Program (`main.py`)

`main.py` coordinates argument parsing, initial condition loading/generation, simulation solving, result persistence, and optional video rendering.

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--processor` | string | **Required** | Choice of back-end: `cpu-serial`, `cpu-parallel`, or `gpu`. |
| `--n-bodies` | int | None | Number of random particles to generate (mutually exclusive with `--initial-condition`). |
| `--initial-condition` | path | None | Path to `.npy` file of shape `(N, 2)` with initial coordinates. |
| `--dt` | float | **Required** | Integrator timestep $\Delta t$ (e.g. `0.05`). |
| `--n-timesteps` | int | **Required** | Total integration steps $T$ (e.g. `1000`). |
| `--n-cores` | int | None | Number of CPU cores (required if `--processor cpu-parallel`). |
| `--render` | string | None | Optional renderer: `discrete` (scatter) or `density` (heatmap). |
| `--render-extent` | string | `both` | Extents to render: `both` (2 videos), `full`, or `initial`. |

---

### Sample Usages & Outputs (`main.py`)

#### Example A: Run Serial Simulation (No Video)
Runs a 400-body simulation for 500 timesteps:
```bash
python main.py --processor cpu-serial --n-bodies 400 --dt 0.05 --n-timesteps 500
```

**Output:**
```
Results saved → results/20261116_203512_a8b9c1d2.npy

────────────────────────────────────────────────────
 Simulation Performance  [CPUSerialModel]
────────────────────────────────────────────────────
 Particles (N)       : 400
 Timesteps (T)       : 500 (dt = 0.05)
 Total wall time     : 7.2140 s
 CPU time            : 7.2110 s (100.0% CPU)
 Time per step       : 14.428 ms
 Simulation rate     : 69.3 steps/s
 Pairwise throughput: 5.53 M pairs/s
 Peak heap memory    : 9.78 MB
 Process RSS memory  : 68.20 MB
────────────────────────────────────────────────────
```

#### Example B: Run Simulation and Generate Both Videos
Runs 300 bodies with `--render discrete` to produce both the **full extent** and **zoomed initial extent (+20%)** videos:
```bash
python main.py --processor cpu-serial --n-bodies 300 --dt 0.05 --n-timesteps 300 --render discrete
```

**Output:**
```
Results saved → results/20261116_204010_e4f5a6b7.npy

────────────────────────────────────────────────────
 Simulation Performance  [CPUSerialModel]
────────────────────────────────────────────────────
 Particles (N)       : 300
 Timesteps (T)       : 300 (dt = 0.05)
 Total wall time     : 2.4510 s
 CPU time            : 2.4490 s (99.9% CPU)
 Time per step       : 8.170 ms
 Simulation rate     : 122.4 steps/s
 Pairwise throughput: 5.49 M pairs/s
 Peak heap memory    : 5.52 MB
 Process RSS memory  : 65.40 MB
────────────────────────────────────────────────────

Rendering 301 frames (full extent) → renders/20261116_204010_e4f5a6b7_full.mp4 …
Video saved → renders/20261116_204010_e4f5a6b7_full.mp4
Rendering 301 frames (initial extent + 20%) → renders/20261116_204010_e4f5a6b7_initial.mp4 …
Video saved → renders/20261116_204010_e4f5a6b7_initial.mp4
```

---

## 4. How to Use the Benchmark Program (`bench.py`)

`bench.py` analyzes how the computational runtime scales with particle count $N$. It performs log-log power-law regression to measure the empirical exponent $b$ in $T(N) = a \cdot N^b$, compares against theoretical $O(N^2)$, and plots a 4-panel performance graph.

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--processor` | string | `cpu-serial` | Solver back-end to test (`cpu-serial`, `cpu-parallel`, `gpu`). |
| `--n-bodies` | ints | `50 100 200 400 600 800 1000` | List of particle counts to benchmark. |
| `--n-timesteps` | int | `100` | Number of timesteps per trial. |
| `--dt` | float | `0.05` | Integration timestep size. |
| `--repeats` | int | `3` | Number of repetitions per $N$ for averaging. |
| `--warmup` | int | `1` | Number of untimed warmup iterations per $N$. |
| `--output-plot` | path | `benchmarks/complexity_scaling.png` | Destination image file for plot. |
| `--output-data` | path | `benchmarks/scaling_data.csv` | Destination CSV file for metrics. |

---

### Sample Usages & Outputs (`bench.py`)

#### Example A: Run Standard Scaling Benchmark
```bash
python bench.py --n-bodies 50 100 200 400 800 --n-timesteps 100 --repeats 2
```

**Output:**
```
======================================================================
 Starting N-Body Scaling Benchmark  [cpu-serial]
 Particle counts (N) : [50, 100, 200, 400, 800]
 Timesteps (T)       : 100 (dt = 0.05)
 Repetitions per N   : 2 (warmup = 1)
======================================================================

     N |   Wall Time (s) |    Std (s) |   Time/Step (ms) |   Peak Heap (MB) |   Throughput (M/s)
────────────────────────────────────────────────────────────────────────────────────────────────
    50 |          0.0187 |     0.0020 |            0.187 |             0.20 |               6.62
   100 |          0.0330 |     0.0016 |            0.330 |             0.62 |              15.02
   200 |          0.2508 |     0.0093 |            2.508 |             2.45 |               7.95
   400 |          1.2995 |     0.0809 |           12.995 |             9.78 |               6.16
   800 |          3.8729 |     0.0120 |           38.729 |            39.09 |               8.25
────────────────────────────────────────────────────────────────────────────────────────────────

======================================================================
 Computational Complexity Analysis
======================================================================
 Empirical Model      : T(N) = 4.16e-06 · N^2.07
 Empirical Exponent   : b = 2.07  (R² = 0.9815)
 Theoretical Scaling  : O(N²) quadratic scaling (R² = 0.9903)
 Verdict              : Closely matches theoretical O(N²) all-pairs scaling!
======================================================================

Benchmark CSV data saved → benchmarks/scaling_data.csv
Benchmark plot successfully saved → benchmarks/complexity_scaling.png
```

#### Generated Benchmark Artifacts:
1. **Plot (`benchmarks/complexity_scaling.png`)**:
   - **Top-left**: Linear scaling $T(N)$ vs $N$ with empirical curve fit vs theoretical $O(N^2)$.
   - **Top-right**: Log-log scaling $\log T$ vs $\log N$ with empirical slope ($b = 2.07$) vs theoretical slope ($b = 2.00$).
   - **Bottom-left**: Pairwise throughput (million interaction pairs / sec) vs $N$.
   - **Bottom-right**: Peak heap memory footprint (MB) vs $N$.
2. **Data (`benchmarks/scaling_data.csv`)**:
   - Full tabular breakdown of mean and std dev across runs.

---

## 5. How to Use the Renderer (`renderers.py`)

`renderers.py` can be used directly from the command line to re-render any existing simulation result (`.npy`) without having to re-run the physics simulation.

### Dual Extent Support
Every render generates two perspectives by default:
1. **Full Extent (`_full.mp4`)**:
   Encompasses all particles across all simulation timesteps with a 5% border buffer.
2. **Initial Extent + 20% (`_initial.mp4`)**:
   Focuses on the initial particle envelope at $t = 0$, expanded by **20% on all sides**:
   $$x \in [x_{\min}^{(0)} - 0.20 \Delta x^{(0)},\; x_{\max}^{(0)} + 0.20 \Delta x^{(0)}]$$
   $$y \in [y_{\min}^{(0)} - 0.20 \Delta y^{(0)},\; y_{\max}^{(0)} + 0.20 \Delta y^{(0)}]$$

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input`, `-i` | path | **Required** | Path to `.npy` result file (shape `2×N×T`). |
| `--render`, `-r` | string | **Required** | `discrete` (scatter particles) or `density` (2D heatmap). |
| `--output`, `-o` | path | `renders/<input_stem>.mp4` | Base output path. Automatically appends `_full.mp4` and `_initial.mp4` when extent is `both`. |
| `--extent`, `-e` | string | `both` | Extent to render: `both` (default, 2 videos), `full`, or `initial`. |
| `--bins` | int | `64` | Number of histogram bins for the `density` renderer. |
| `--fps` | int | `30` | Frames per second. |
| `--dpi` | int | `100` | Matplotlib figure resolution. |

---

### Sample Usages & Outputs (`renderers.py`)

#### Example A: Render Heatmap Density Videos (Both Extents)
```bash
python renderers.py --input results/20261116_203301_052b4958.npy --render density
```

**Output:**
```
Rendering 16 frames (full extent) → renders/20261116_203301_052b4958_full.mp4 …
Video saved → renders/20261116_203301_052b4958_full.mp4
Rendering 16 frames (initial extent + 20%) → renders/20261116_203301_052b4958_initial.mp4 …
Video saved → renders/20261116_203301_052b4958_initial.mp4
```

#### Example B: Render Only the Zoomed Initial Extent as Particles
```bash
python renderers.py -i results/20261116_203301_052b4958.npy -r discrete --extent initial -o renders/zoom.mp4
```

**Output:**
```
Rendering 16 frames (initial extent + 20%) → renders/zoom.mp4 …
Video saved → renders/zoom.mp4
```

---

## 6. Guide for Group Members Implementing New Solvers

To implement a new parallel CPU or GPU solver, open [models.py](models.py) and fill in the `_run_solver(self)` method in the respective class:

- **Parallel CPU**: Edit `CPUParallelModel._run_solver` (e.g. using `multiprocessing`, `numba.prange`, or `joblib`).
- **GPU**: Edit `GPUModel._run_solver` (e.g. using `CuPy`, PyCUDA, or Numba CUDA).

Because all subclasses inherit from `GenericModel`, initial condition generation, timing, memory monitoring, result saving, benchmarking, and video rendering will automatically work without any extra boilerplate!
