"""
N-body simulation renderers.

GenericRenderer handles loading simulation data and encoding video outputs.
Subclasses produce individual frames:
  - DiscreteRenderer  : scatter-plot of individual particles
  - DensityRenderer   : 2-D histogram / heatmap of particle density

By default, visualization produces two renders:
  1. Full extent (all particles across all timesteps + 5% buffer)
  2. Initial extent (particles at t=0 + 20% margin on all sides)

Can be used from main.py or as a standalone CLI::

    # Produces both full and initial extent renders:
    python renderers.py --input results/20261116_200000_abc12345.npy --render discrete

    # Choose specific extent:
    python renderers.py --input results/20261116_200000_abc12345.npy --render density --extent initial
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from abc import ABC, abstractmethod


# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_FPS = 30
DEFAULT_BITRATE = 10_000
DEFAULT_DPI = 100
DEFAULT_BINS = 64          # for density heatmap


class GenericRenderer(ABC):
    """
    Base renderer: loads data, computes extents, drives the frame loop, encodes video.

    Parameters
    ----------
    data : np.ndarray, shape (2, N, T+1)
        Position array produced by a solver.
    output : str
        Base path to the output video file (e.g. 'renders/simulation.mp4').
    extent : str
        Which extent(s) to render: 'both' (default), 'full', or 'initial'.
    xlim, ylim : tuple[float, float] or None
        Optional custom coordinate limits. If supplied, overrides extent modes.
    fps : int
        Frames per second.
    bitrate : float
        Video bitrate.
    dpi : int
        Figure DPI.
    """

    def __init__(
        self,
        data: np.ndarray,
        output: str = "renders/simulation.mp4",
        *,
        extent: str = "both",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        fps: int = DEFAULT_FPS,
        bitrate: float = DEFAULT_BITRATE,
        dpi: int = DEFAULT_DPI,
    ):
        self.data = data                     # (2, N, T+1)
        self.n_bodies = data.shape[1]
        self.n_frames = data.shape[2]
        self.output = output
        self.extent = extent
        self.fps = fps
        self.bitrate = bitrate
        self.dpi = dpi

        # 1. Full extent: over all particles across all timesteps (+ 5% buffer)
        self.xlim_full, self.ylim_full = self.compute_full_extent(data, buffer_ratio=0.05)

        # 2. Initial extent: particle extent at t=0 + 20% on all sides
        self.xlim_initial, self.ylim_initial = self.compute_initial_extent(data, buffer_ratio=0.20)

        # Custom bounds if explicitly given
        self.custom_xlim = xlim
        self.custom_ylim = ylim

    # ── Extent calculation helpers ───────────────────────────────────────

    @staticmethod
    def compute_full_extent(data: np.ndarray, buffer_ratio: float = 0.05) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute bounds encompassing all particles across all timesteps with a buffer."""
        all_x = data[0].ravel()
        all_y = data[1].ravel()
        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min, y_max = float(all_y.min()), float(all_y.max())
        span_x = x_max - x_min
        span_y = y_max - y_min
        buf_x = buffer_ratio * span_x if span_x > 0 else 1.0
        buf_y = buffer_ratio * span_y if span_y > 0 else 1.0
        return (x_min - buf_x, x_max + buf_x), (y_min - buf_y, y_max + buf_y)

    @staticmethod
    def compute_initial_extent(data: np.ndarray, buffer_ratio: float = 0.20) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute initial particle bounds (t=0) expanded by buffer_ratio (default 20%) on all sides."""
        init_x = data[0, :, 0]
        init_y = data[1, :, 0]
        x0_min, x0_max = float(init_x.min()), float(init_x.max())
        y0_min, y0_max = float(init_y.min()), float(init_y.max())
        span_x = x0_max - x0_min
        span_y = y0_max - y0_min
        pad_x = buffer_ratio * span_x if span_x > 0 else 1.0
        pad_y = buffer_ratio * span_y if span_y > 0 else 1.0
        return (x0_min - pad_x, x0_max + pad_x), (y0_min - pad_y, y0_max + pad_y)

    @property
    def xlim(self) -> tuple[float, float]:
        """Backwards compatibility: returns full extent xlim."""
        return self.xlim_full

    @property
    def ylim(self) -> tuple[float, float]:
        """Backwards compatibility: returns full extent ylim."""
        return self.ylim_full

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args, data: np.ndarray | None = None) -> "GenericRenderer":
        """
        Build the right renderer subclass from the parsed CLI ``args``.

        If *data* is not supplied it is loaded from ``args.input``.
        """
        if data is None:
            data = np.load(args.input)

        output = getattr(args, "output", None)
        if not output:
            input_stem = os.path.splitext(os.path.basename(args.input))[0] if getattr(args, "input", None) else "simulation"
            output = f"renders/{input_stem}.mp4"

        extent = getattr(args, "extent", "both")
        fps = getattr(args, "fps", DEFAULT_FPS)
        bitrate = getattr(args, "bitrate", DEFAULT_BITRATE)
        dpi = getattr(args, "dpi", DEFAULT_DPI)

        renderer_map: dict[str, type[GenericRenderer]] = {
            "discrete": DiscreteRenderer,
            "density": DensityRenderer,
        }
        subclass = renderer_map.get(args.render)
        if subclass is None:
            raise ValueError(f"Unknown renderer: {args.render}")

        kwargs = dict(
            output=output,
            extent=extent,
            fps=fps,
            bitrate=bitrate,
            dpi=dpi,
        )
        if args.render == "density" and hasattr(args, "bins"):
            kwargs["bins"] = args.bins

        return subclass(data, **kwargs)

    @classmethod
    def renderer(cls, args, data: np.ndarray | None = None) -> "GenericRenderer":
        """Alias for from_args."""
        return cls.from_args(args, data=data)

    # ── Public interface ─────────────────────────────────────────────────

    def render(self) -> list[str]:
        """
        Produce video file(s) according to ``self.extent``.

        Returns
        -------
        list[str]
            Paths of the saved video files.
        """
        root, ext = os.path.splitext(self.output)
        if not ext:
            ext = ".mp4"

        clean_root = root
        for suffix in ("_full", "_initial"):
            if clean_root.endswith(suffix):
                clean_root = clean_root[:-len(suffix)]
                break

        # If custom limits were provided, render just that single view
        if self.custom_xlim is not None and self.custom_ylim is not None:
            out_custom = self.output
            self._render_single(self.custom_xlim, self.custom_ylim, out_custom, label="custom")
            return [out_custom]

        rendered_files = []

        if self.extent in ("full", "both"):
            out_full = f"{clean_root}_full{ext}" if self.extent == "both" else self.output
            self._render_single(self.xlim_full, self.ylim_full, out_full, label="full extent")
            rendered_files.append(out_full)

        if self.extent in ("initial", "both"):
            out_init = f"{clean_root}_initial{ext}" if self.extent == "both" else self.output
            self._render_single(self.xlim_initial, self.ylim_initial, out_init, label="initial extent + 20%")
            rendered_files.append(out_init)

        return rendered_files

    # ── Internal helpers ─────────────────────────────────────────────────

    def _setup_figure(self, xlim: tuple[float, float], ylim: tuple[float, float]):
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        return fig, ax

    def _render_single(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        output_path: str,
        label: str = "",
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fig, ax = self._setup_figure(xlim, ylim)
        artists = self._init_frame(ax, xlim, ylim)

        def _update(frame: int):
            return self._update_frame(ax, artists, frame, xlim, ylim)

        writer = animation.FFMpegWriter(fps=self.fps, bitrate=self.bitrate)
        ani = animation.FuncAnimation(
            fig, _update, frames=self.n_frames, interval=1000 / self.fps, blit=False
        )
        desc = f" ({label})" if label else ""
        print(f"Rendering {self.n_frames} frames{desc} → {output_path} …")
        ani.save(output_path, writer=writer, dpi=self.dpi)
        plt.close(fig)
        print(f"Video saved → {output_path}")
        return output_path

    # ── To be implemented by subclasses ──────────────────────────────────

    @abstractmethod
    def _init_frame(self, ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> object:
        """Set up the first frame and return artist handles."""
        ...

    @abstractmethod
    def _update_frame(
        self,
        ax,
        artists,
        frame: int,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> object:
        """Update the artists for the given *frame* index."""
        ...


# ═════════════════════════════════════════════════════════════════════════════
#  Concrete renderers
# ═════════════════════════════════════════════════════════════════════════════

class DiscreteRenderer(GenericRenderer):
    """Render each particle as an individual dot (scatter plot)."""

    def _init_frame(self, ax, xlim, ylim):
        scatter = ax.scatter(
            self.data[0, :, 0],
            self.data[1, :, 0],
            s=4,
            c="white",
            edgecolors="none",
        )
        return scatter

    def _update_frame(self, ax, scatter, frame, xlim, ylim):
        scatter.set_offsets(self.data[:, :, frame].T)
        return (scatter,)


class DensityRenderer(GenericRenderer):
    """Render a 2-D histogram heatmap of particle density."""

    def __init__(self, *args, bins: int = DEFAULT_BINS, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins

    def _init_frame(self, ax, xlim, ylim):
        H, xedges, yedges = np.histogram2d(
            self.data[0, :, 0],
            self.data[1, :, 0],
            bins=self.bins,
            range=[list(xlim), list(ylim)],
        )
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=[*xlim, *ylim],
            cmap="inferno",
            aspect="equal",
            interpolation="gaussian",
        )
        return im

    def _update_frame(self, ax, im, frame, xlim, ylim):
        H, _, _ = np.histogram2d(
            self.data[0, :, frame],
            self.data[1, :, frame],
            bins=self.bins,
            range=[list(xlim), list(ylim)],
        )
        im.set_data(H.T)
        vmax = H.max() if H.max() > 0 else 1
        im.set_clim(vmin=0, vmax=vmax)
        return (im,)


# ═════════════════════════════════════════════════════════════════════════════
#  Standalone CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an N-body simulation result to video (full extent & initial extent + 20%)."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a .npy result file (shape 2×N×T).",
    )
    parser.add_argument(
        "--render", "-r",
        required=True,
        choices=["discrete", "density"],
        help="Renderer type.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Base output video path (default: renders/<input_stem>.mp4). When extent is 'both', appends _full.mp4 and _initial.mp4.",
    )
    parser.add_argument(
        "--extent", "-e",
        choices=["both", "full", "initial"],
        default="both",
        help="Extent to render: 'both' (default, produces 2 videos), 'full', or 'initial' (initial extent + 20%%).",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--bitrate", type=float, default=DEFAULT_BITRATE)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--bins", type=int, default=DEFAULT_BINS,
        help="Histogram bins for density renderer (default: 64).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    renderer = GenericRenderer.from_args(args)
    renderer.render()


if __name__ == "__main__":
    main()
