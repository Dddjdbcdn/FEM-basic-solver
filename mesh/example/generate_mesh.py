"""Regenerate the included example without requiring a mesh generator."""

from pathlib import Path

import meshio
import numpy as np

from fem.input import box_mesh

if __name__ == "__main__":
    mesh = box_mesh()
    path = Path(__file__).with_name("cantilever.msh")
    meshio.write(
        path,
        meshio.Mesh(
            mesh.points,
            [("tetra", mesh.tetra)],
            cell_data={
                "gmsh:physical": [np.ones(len(mesh.tetra), dtype=np.int32)],
                "gmsh:geometrical": [np.ones(len(mesh.tetra), dtype=np.int32)],
            },
            field_data={"solid": np.array([1, 3])},
        ),
        file_format="gmsh22",
        binary=False,
    )
    print(f"Wrote {path}: {len(mesh.points)} nodes, {len(mesh.tetra)} tetrahedra")
