import copy
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from fem.construct import build_mesh, construct, evaluate
from fem.input import load_case, load_mesh, surface_faces
from fem.solver import solve


@pytest.fixture
def table():
    path = Path(__file__).parents[1] / "mesh" / "table.json"
    return json.loads(path.read_text())


def test_table_volume_surface_and_connectivity(table):
    mesh, parts, groups = build_mesh(table)
    xyz = mesh.points[mesh.tetra]
    volumes = np.linalg.det(xyz[:, 1:] - xyz[:, :1]) / 6
    assert np.all(volumes > 0)
    assert len(np.unique(mesh.tetra)) == len(mesh.points)
    assert_allclose(volumes.sum(), 1.2 * 0.08 * 0.8 + 4 * 0.08 * 0.72 * 0.08)
    assert_allclose(volumes[parts == 0].sum(), 1.2 * 0.08 * 0.8)
    assert_allclose(mesh.points[mesh.groups["feet"], 1], 0)
    assert_allclose(mesh.points[mesh.groups["load_surface"], 1], 0.8)

    # Shared triangular faces must connect every element, even at partial face joints.
    neighbours = []
    seen = {}
    for element, nodes in enumerate(mesh.tetra):
        for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted([nodes[a], nodes[b], nodes[c]]))
            if key in seen:
                neighbours.append((seen[key], element))
            else:
                seen[key] = element
    rows, columns = np.array(neighbours).T
    graph = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(mesh.tetra), len(mesh.tetra))
    )
    count, labels = connected_components(graph, directed=False)
    assert count == 1

    faces, owners = surface_faces(mesh)
    triangles = mesh.points[faces]
    area = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        ).sum()
        / 2
    )
    top_area = 2 * (1.2 * 0.8 + 1.2 * 0.08 + 0.8 * 0.08)
    leg_area = 2 * (0.08 * 0.72 + 0.08 * 0.08 + 0.72 * 0.08)
    assert_allclose(area, top_area + 4 * leg_area - 2 * 4 * 0.08**2)


def test_export_roundtrip_and_solve(table, tmp_path):
    source = tmp_path / "table.json"
    source.write_text(json.dumps(table))
    original = construct(source, tmp_path / "built" / "table.msh")
    mesh = load_mesh(tmp_path / "built" / "table.msh")
    assert set(mesh.parts) == set(original.parts)
    assert set(mesh.groups) == set(original.groups)
    for name in mesh.groups:
        assert_allclose(mesh.groups[name], original.groups[name])
    for name in mesh.parts:
        assert_allclose(mesh.parts[name], original.parts[name])

    mesh, material, boundary = load_case(tmp_path / "built" / "table.case.json")
    result = solve(mesh, material, boundary)
    assert np.all(np.isfinite(result.stress))
    assert result.displacement[:, 1].min() < 0
    assert_allclose(result.displacement[mesh.groups["feet"]], 0)
    assert_allclose(result.reaction.sum(axis=0), [0, 1000, 0], atol=1e-5)
    assert_allclose(
        np.cross(mesh.points, boundary.force + result.reaction).sum(axis=0), 0, atol=1e-5
    )


def test_parameter_edit_changes_geometry(table):
    table["parameters"]["width"] = 1.6
    mesh, parts, groups = build_mesh(table)
    assert_allclose(mesh.points[:, 0].max(), 1.6)
    right = mesh.points[mesh.groups["front_right_leg"]]
    assert_allclose(right[:, 0].min(), 1.6 - 0.04 - 0.08)
    assert_allclose(right[:, 0].max(), 1.6 - 0.04)


@pytest.mark.parametrize(
    "second, message",
    [
        ({"origin": [0.5, 0, 0], "size": [1, 1, 1]}, "overlap"),
        ({"origin": [2, 0, 0], "size": [1, 1, 1]}, "connect"),
        ({"origin": [1, 1, 0], "size": [1, 1, 1]}, "edge or point"),
        ({"origin": [1, 1, 1], "size": [1, 1, 1]}, "edge or point"),
        ({"origin": [1, 0, 0], "size": [0, 1, 1]}, "positive"),
    ],
)
def test_invalid_box_connections_fail_before_meshing(second, message):
    data = {
        "mesh_size": 0.25,
        "parts": [
            {"name": "a", "origin": [0, 0, 0], "size": [1, 1, 1]},
            {"name": "b", **second},
        ],
    }
    with pytest.raises(ValueError, match=message):
        build_mesh(data)


def test_unknown_face_and_empty_groups_rejected(table):
    data = copy.deepcopy(table)
    data["groups"]["feet"]["face"] = "bottom"
    with pytest.raises(ValueError, match="Face must"):
        build_mesh(data)
    data = copy.deepcopy(table)
    data["groups"] = {"joint": {"parts": ["front_left_leg"], "face": "ymax"}}
    with pytest.raises(ValueError, match="no exterior"):
        build_mesh(data)


def test_arithmetic_expressions_cannot_execute_python():
    assert evaluate("(width - 2 * inset) / 2", {"width": 1.2, "inset": 0.1}) == 0.5
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('echo unexpected')", {})
