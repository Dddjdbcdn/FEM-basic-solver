import numpy as np
from numpy.testing import assert_allclose

from fem.input import box_mesh, build_boundary, select_nodes


def test_total_force_is_independent_of_selection_size():
    for div in ((1, 1, 1), (2, 3, 4)):
        mesh = box_mesh(divisions=div)
        bc = build_boundary(
            mesh,
            [],
            [
                {"select": {"axis": "x", "at": "max"}, "force": [100, -200, 300]},
            ],
        )
        assert_allclose(bc.force.sum(axis=0), [100, -200, 300])


def test_selectors_and_partial_constraints():
    mesh = box_mesh(divisions=(2, 2, 2))
    ids = select_nodes(mesh, {"axis": "x", "at": 0.5})
    assert len(ids) == 9
    box = select_nodes(mesh, {"box": [[0.5, 0, 0], [0.5, 0.1, 0.1]]})
    assert_allclose(ids, box)
    mesh.groups["middle"] = ids
    assert_allclose(select_nodes(mesh, {"group": "middle"}), ids)
    assert_allclose(select_nodes(mesh, {"nodes": [2, 0, 2]}), [0, 2])
    bc = build_boundary(
        mesh,
        [
            {"select": {"group": "middle"}, "displacement": [0, None, 0.001]},
        ],
        [{"select": {"nodes": [0, 2]}, "force": [1, 2, 3], "distribution": "per_node"}],
    )
    assert np.all(bc.fixed[ids][:, [0, 2]])
    assert not np.any(bc.fixed[:, 1])
    assert_allclose(bc.values[ids, 2], 0.001)
    assert_allclose(bc.force.sum(axis=0), [2, 4, 6])
