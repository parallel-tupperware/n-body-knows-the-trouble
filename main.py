#!/usr/bin/env python3
"""
N-body simulation – main entry point.

Usage examples
--------------
# 500 random bodies, serial CPU, 2000 steps, dt=0.05
python main.py --processor cpu-serial --n-bodies 500 --dt 0.05 --n-timesteps 2000

# Load initial conditions, parallel CPU on 4 cores, and render both full & initial extents
python main.py --processor cpu-parallel --n-cores 4 \\
               --initial-condition ic.npy --dt 0.01 --n-timesteps 5000 \\
               --render density

# GPU solver with discrete particle rendering
python main.py --processor gpu --n-bodies 1000 --dt 0.02 --n-timesteps 3000 \\
               --render discrete
"""

import os
import argparse
import sys

from models import GenericModel
from renderers import GenericRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="N-body gravitational simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Processor ────────────────────────────────────────────────────────
    parser.add_argument(
        "--processor",
        required=True,
        choices=["cpu-serial", "cpu-parallel", "gpu"],
        help="Solver back-end to use.",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Number of CPU cores (required when processor is cpu-parallel).",
    )

    # ── Initial conditions ───────────────────────────────────────────────
    ic_group = parser.add_mutually_exclusive_group(required=True)
    ic_group.add_argument(
        "--initial-condition",
        type=str,
        default=None,
        help="Path to a .npy file with shape (N, 2) of initial positions.",
    )
    ic_group.add_argument(
        "--n-bodies",
        type=int,
        default=None,
        help="Number of bodies to generate randomly (used when no IC file is given).",
    )

    # ── Integration parameters ───────────────────────────────────────────
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Timestep for the integrator.",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        required=True,
        help="Number of integration timesteps.",
    )

    # ── Rendering (optional) ─────────────────────────────────────────────
    parser.add_argument(
        "--render",
        choices=["discrete", "density"],
        default=None,
        help="If specified, render the result to video after solving (generates full extent & initial extent + 20%%).",
    )
    parser.add_argument(
        "--render-extent",
        choices=["both", "full", "initial"],
        default="both",
        help="Extent to render: 'both' (default, generates 2 videos), 'full', or 'initial' (initial extent + 20%%).",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Extra cross-field validation that argparse can't express."""
    if args.processor == "cpu-parallel" and args.n_cores is None:
        sys.exit("Error: --n-cores is required when --processor is cpu-parallel.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    # ── Build & run model ────────────────────────────────────────────────
    model = GenericModel.from_args(args)
    model.solve(save=True)

    # ── Optional rendering ───────────────────────────────────────────────
    if args.render is not None:
        base_name = (
            os.path.splitext(os.path.basename(model.file))[0]
            if model.file
            else "simulation"
        )
        render_args = argparse.Namespace(
            input=model.file,
            render=args.render,
            output=f"renders/{base_name}.mp4",
            extent=args.render_extent,
        )
        renderer = GenericRenderer.renderer(render_args)
        renderer.render()


if __name__ == "__main__":
    main()
