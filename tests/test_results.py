import meshio
import numpy as np
from numpy.testing import assert_allclose

from fem.input import box_mesh, build_boundary
from fem.results import average_stress, export_result, von_mises
from fem.solver import solve


def test_averaging_preserves_constant_stress_and_averages_tensors_first():
    mesh = box_mesh(divisions=(1, 1, 1))
    stress = np.zeros((len(mesh.tetra), 6))
    stress[:, 0] = np.tile([10, -10], 3)
    averaged = average_stress(mesh, stress, np.ones(len(stress)))
    # Opposite stresses cancel at the two nodes shared by all six elements.
    assert_allclose(averaged[[0, 7]], 0)
    assert_allclose(von_mises(averaged[[0, 7]]), 0)
    assert_allclose(von_mises(stress), 10)
    constant = np.tile([1, 2, 3, 4, 5, 6], (len(stress), 1))
    assert_allclose(
        average_stress(mesh, constant, np.arange(1, len(stress) + 1)),
        np.tile([1, 2, 3, 4, 5, 6], (len(mesh.points), 1)),
    )


def test_vtu_roundtrip_retains_raw_and_averaged_data(tmp_path, material):
    mesh = box_mesh(divisions=(2, 1, 1))
    bc = build_boundary(
        mesh,
        [
            {"select": {"axis": "x", "at": "min"}, "displacement": [0, 0, 0]},
        ],
        [{"select": {"axis": "x", "at": "max"}, "force": [0, 0, -1]}],
    )
    result = solve(mesh, material, bc)
    path = export_result(tmp_path / "nested" / "result.vtu", mesh, result)
    loaded = meshio.read(path)
    assert_allclose(loaded.points, mesh.points)
    assert_allclose(loaded.cells_dict["tetra"], mesh.tetra)
    assert_allclose(loaded.point_data["displacement_m"], result.displacement)
    assert_allclose(loaded.cell_data_dict["von_mises_Pa"]["tetra"], result.von_mises)
    assert_allclose(loaded.point_data["von_mises_averaged_Pa"], result.nodal_von_mises)
    tensor = loaded.cell_data_dict["stress_Pa"]["tetra"].reshape(-1, 3, 3)
    assert_allclose(tensor[:, 0, 1], result.stress[:, 3])
