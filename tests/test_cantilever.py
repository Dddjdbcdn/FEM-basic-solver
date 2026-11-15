import numpy as np
from numpy.testing import assert_allclose

from fem.input import Material, box_mesh, build_boundary
from fem.solver import solve


def test_cantilever_refines_toward_beam_theory_and_balances_load():
    # Slender beam; nu=0 reduces end effects in comparison with Euler-Bernoulli theory.
    length, width, height, force, young = 1.0, 0.1, 0.1, 1000.0, 70e9
    inertia = width * height**3 / 12
    theoretical = force * length**3 / (3 * young * inertia)
    errors = []
    for divisions in ((8, 2, 2), (16, 4, 4), (32, 8, 8)):
        mesh = box_mesh((length, width, height), divisions)
        bc = build_boundary(
            mesh,
            [
                {"select": {"axis": "x", "at": "min"}, "displacement": [0, 0, 0]},
            ],
            [{"select": {"axis": "x", "at": "max"}, "force": [0, 0, -force]}],
        )
        result = solve(mesh, Material(young, 0.0), bc)
        tip = np.isclose(mesh.points[:, 0], length)
        deflection = -result.displacement[tip, 2].mean()
        errors.append(abs(deflection / theoretical - 1))
        assert_allclose(result.reaction.sum(axis=0), [0, 0, force], atol=2e-5)
        applied_moment = np.cross(mesh.points, bc.force).sum(axis=0)
        reaction_moment = np.cross(mesh.points, result.reaction).sum(axis=0)
        assert_allclose(applied_moment + reaction_moment, 0, atol=2e-5)
        assert result.relative_residual < 1e-8
        assert result.strain_energy > 0
        assert_allclose(2 * result.strain_energy, np.sum(bc.force * result.displacement), rtol=1e-8)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.15
