import numpy as np
from numpy.testing import assert_allclose

from fem.input import BoundaryConditions, box_mesh, build_boundary
from fem.solver import solve


def test_constant_strain_patch_with_free_interior_node(material):
    mesh = box_mesh(size=(1.2, 0.8, 0.6), divisions=(2, 2, 2))
    a = np.array([[1e-4, 2e-4, -3e-4], [0, -2e-4, 1e-4], [2e-4, 0, 3e-4]])
    expected_u = mesh.points @ a.T + [1e-5, -2e-5, 3e-5]
    on_surface = np.any(
        np.isclose(mesh.points, 0) | np.isclose(mesh.points, [1.2, 0.8, 0.6]), axis=1
    )
    fixed = np.repeat(on_surface[:, None], 3, axis=1)
    result = solve(
        mesh, material, BoundaryConditions(np.zeros_like(mesh.points), fixed, expected_u)
    )
    assert np.count_nonzero(~on_surface) == 1
    assert_allclose(result.displacement, expected_u, rtol=1e-11, atol=1e-16)
    eps = (a + a.T) / 2
    lam, mu = material.lame
    sigma = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
    components = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    expected_stress = np.tile([sigma[i, j] for i, j in components], (len(mesh.tetra), 1))
    assert_allclose(result.stress, expected_stress, rtol=1e-11, atol=1e-6)
    assert_allclose(result.reaction.sum(axis=0), 0, atol=1e-7)
    assert_allclose(result.strain_energy, 0.5 * np.sum(eps * sigma) * 1.2 * 0.8 * 0.6)


def test_uniaxial_prescribed_displacement_and_poisson_contraction(material):
    mesh = box_mesh(size=(2, 0.4, 0.3), divisions=(4, 2, 2))
    bc = build_boundary(
        mesh,
        [
            {"select": {"axis": "x", "at": "min"}, "displacement": [0, None, None]},
            {"select": {"axis": "y", "at": "min"}, "displacement": [None, 0, None]},
            {"select": {"axis": "z", "at": "min"}, "displacement": [None, None, 0]},
            {"select": {"axis": "x", "at": "max"}, "displacement": [0.002, None, None]},
        ],
        [],
    )
    result = solve(mesh, material, bc)
    assert_allclose(result.displacement, mesh.points * [0.001, -0.0003, -0.0003], atol=1e-15)
    assert_allclose(result.stress[:, 0], material.young * 0.001, rtol=1e-11)
    assert_allclose(result.stress[:, 1:], 0, atol=1e-5)
    right = np.isclose(mesh.points[:, 0], 2)
    assert_allclose(result.reaction[right, 0].sum(), material.young * 0.001 * 0.4 * 0.3)


def test_zero_load_and_fully_prescribed_cases(material):
    mesh = box_mesh(divisions=(1, 1, 1))
    for selection in ({"axis": "x", "at": "min"}, {"nodes": list(range(len(mesh.points)))}):
        bc = build_boundary(mesh, [{"select": selection, "displacement": [0, 0, 0]}], [])
        result = solve(mesh, material, bc)
        assert_allclose(result.displacement, 0)
        assert_allclose(result.stress, 0)
        assert result.relative_residual == 0
