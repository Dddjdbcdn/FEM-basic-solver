"""Stress invariants, documented nodal averaging, and portable result export."""

from pathlib import Path

import meshio
import numpy as np


def von_mises(stress):
    xx, yy, zz, xy, yz, xz = stress.T
    normal = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
    shear = 3 * (xy**2 + yz**2 + xz**2)
    return np.sqrt(normal + shear)


def average_stress(mesh, stress, volumes):
    """Average tensors by incident element volume, then compute invariants later."""
    total = np.zeros((len(mesh.points), 6))
    weight = np.zeros(len(mesh.points))
    for element, nodes in enumerate(mesh.tetra):
        for node in nodes:
            total[node] += stress[element] * volumes[element]
            weight[node] += volumes[element]
    return total / weight[:, None]


def stress_tensor(components):
    """Flatten symmetric tensors to VTK's nine-component row-major layout."""
    tensor = np.empty((len(components), 3, 3))
    for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))):
        tensor[:, i, j] = tensor[:, j, i] = components[:, k]
    return tensor.reshape(-1, 9)


def export_result(path, mesh, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        path,
        meshio.Mesh(
            mesh.points,
            [("tetra", mesh.tetra)],
            point_data={
                "displacement_m": result.displacement,
                "displacement_magnitude_m": np.linalg.norm(result.displacement, axis=1),
                "reaction_N": result.reaction,
                "stress_averaged_Pa": stress_tensor(result.nodal_stress),
                "von_mises_averaged_Pa": result.nodal_von_mises,
            },
            cell_data={
                "stress_Pa": [stress_tensor(result.stress)],
                "strain": [stress_tensor(result.strain)],
                "von_mises_Pa": [result.von_mises],
                "volume_m3": [result.volumes],
            },
        ),
    )
    return path
