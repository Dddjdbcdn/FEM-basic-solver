"""Run with: python -m fem mesh/example/cantilever.json"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .input import load_case, load_mesh
from .results import export_result
from .solver import solve
from .viewer import view


def main():
    parser = argparse.ArgumentParser(description="3D linear elastic FEM")
    parser.add_argument("case", type=Path)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--up", choices=["x", "y", "z"], default="z")
    args = parser.parse_args()

    if args.mesh_only:
        mesh = load_mesh(args.case)
        result = None
        boundary = None
    else:
        start = perf_counter()
        mesh, material, boundary = load_case(args.case)
        result = solve(mesh, material, boundary)
        output = args.output or Path("output") / f"{args.case.stem}.vtu"
        export_result(output, mesh, result)

        print(f"Solved {len(mesh.tetra):,} tetrahedra in {perf_counter() - start:.3f} s")
        print(f"Max displacement: {np.linalg.norm(result.displacement, axis=1).max():.6e} m")
        print(f"Max von Mises stress: {result.von_mises.max():.6e} Pa")
        print(f"Results: {output}")

    if not args.no_view:
        view(
            mesh,
            result,
            boundary,
            title=args.case.stem,
            screenshot=args.screenshot,
            frames=args.frames,
            up=args.up,
        )


if __name__ == "__main__":
    main()
