"""Case input: data containers, tetrahedral meshes, materials, supports, and loads."""

import json
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import meshio
import numpy as np

# Data containers


@dataclass
class Material:
    young: float
    poisson: float

    @property
    def lame(self) -> tuple[float, float]:
        e, nu = self.young, self.poisson
        return e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))


@dataclass
class Mesh:
    points: np.ndarray  # (nodes, 3), metres
    tetra: np.ndarray  # (elements, 4), zero-based node IDs
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    parts: dict[str, np.ndarray] = field(default_factory=dict)  # Named element sets


@dataclass
class BoundaryConditions:
    force: np.ndarray  # (nodes, 3), newtons
    fixed: np.ndarray  # (nodes, 3), bool
    values: np.ndarray  # (nodes, 3), prescribed displacement in metres


@dataclass
class Result:
    displacement: np.ndarray  # (nodes, 3), metres
    reaction: np.ndarray  # (nodes, 3), nonzero only on constrained DOFs
    strain: np.ndarray  # (elements, 6), tensor shear, xx yy zz xy yz xz
    stress: np.ndarray  # (elements, 6), pascals, same component order
    von_mises: np.ndarray  # (elements,), pascals
    nodal_stress: np.ndarray  # volume-weighted stress tensor average
    nodal_von_mises: np.ndarray  # von Mises of the averaged stress tensor
    volumes: np.ndarray  # (elements,), cubic metres
    relative_residual: float
    strain_energy: float


# Mesh loading and geometry


def load_mesh(path):
    raw = meshio.read(path)
    points = np.array(raw.points, dtype=np.float64)
    tetra = np.array(raw.cells_dict["tetra"], dtype=np.int32)
    groups = {}

    for name, nodes in raw.point_sets.items():
        groups[name] = np.array(nodes, dtype=int)

    for name, selections in raw.cell_sets.items():
        nodes = []
        for block, selected in zip(raw.cells, selections):
            if selected is not None:
                nodes.extend(block.data[selected].ravel())
        groups[name] = np.unique(nodes).astype(int)

    # Gmsh physical groups associate a name with a tag and a geometric dimension.
    for name, (tag, dimension) in raw.field_data.items():
        nodes = []
        for block, tags in zip(raw.cells, raw.cell_data.get("gmsh:physical", [])):
            if block.dim == dimension:
                nodes.extend(block.data[tags == tag].ravel())
        groups[name] = np.unique(nodes).astype(int)

    parts = {}
    tags = raw.cell_data_dict.get("gmsh:physical", {}).get("tetra")
    if tags is not None:
        for name, (tag, dimension) in raw.field_data.items():
            if dimension == 3:
                parts[name] = np.flatnonzero(tags == tag)
    return Mesh(points, tetra, groups, parts)


def surface_faces(mesh):
    # A face seen once is exterior. A face shared by two tetrahedra is interior.
    local_faces = [(1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3)]
    counts = {}
    face_info = {}
    for element, nodes in enumerate(mesh.tetra):
        for a, b, c, opposite in local_faces:
            face = [nodes[a], nodes[b], nodes[c]]
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
            face_info[key] = (face, element, nodes[opposite])

    faces = []
    owners = []
    for key, (face, element, opposite) in face_info.items():
        if counts[key] == 1:
            a, b, c = mesh.points[face]
            normal = np.cross(b - a, c - a)
            if np.dot(normal, mesh.points[opposite] - a) > 0:
                face[1], face[2] = face[2], face[1]
            faces.append(face)
            owners.append(element)

    return np.array(faces, dtype=np.int32), np.array(owners)


def box_mesh(size=(1.0, 0.1, 0.1), divisions=(40, 8, 8)) -> Mesh:
    """Conforming six-tet-per-cell box, useful for examples and verification."""
    size = np.asarray(size, dtype=np.float64)
    divisions = np.asarray(divisions)
    axes = [np.linspace(0, size[i], divisions[i] + 1) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    index = np.arange(len(points)).reshape(tuple(divisions + 1))
    tetra = []
    for origin in np.ndindex(tuple(divisions)):
        for order in permutations(range(3)):
            node = np.array(origin)
            cell = [index[tuple(node)]]
            for axis in order:
                node[axis] += 1
                cell.append(index[tuple(node)])
            tetra.append(cell)
    return Mesh(points, np.array(tetra, dtype=np.int32))


# Node selections, supports, loads, and JSON input


def select_nodes(mesh, selection):
    if "nodes" in selection:
        return np.unique(selection["nodes"]).astype(int)
    if "group" in selection:
        return mesh.groups[selection["group"]]

    tolerance = selection.get("tol", 1e-8 * np.ptp(mesh.points, axis=0).max())
    if "box" in selection:
        lower, upper = np.array(selection["box"])
        inside = (mesh.points >= lower - tolerance) & (mesh.points <= upper + tolerance)
        return np.flatnonzero(inside.all(axis=1))

    axis = "xyz".index(selection["axis"])
    coordinates = mesh.points[:, axis]
    at = selection["at"]
    if at == "min":
        at = coordinates.min()
    elif at == "max":
        at = coordinates.max()
    return np.flatnonzero(np.abs(coordinates - at) <= tolerance)


def build_boundary(mesh, constraints, loads):
    force = np.zeros_like(mesh.points)
    fixed = np.zeros(mesh.points.shape, dtype=bool)
    values = np.zeros_like(mesh.points)

    for constraint in constraints:
        nodes = select_nodes(mesh, constraint["select"])
        for component, value in enumerate(constraint["displacement"]):
            if value is not None:  # null in JSON means this component is free.
                fixed[nodes, component] = True
                values[nodes, component] = value

    for load in loads:
        nodes = select_nodes(mesh, load["select"])
        nodal_force = np.array(load["force"], dtype=float)
        if load.get("distribution", "total") == "total":
            nodal_force /= len(nodes)
        force[nodes] += nodal_force

    return BoundaryConditions(force, fixed, values)


def load_case(path):
    path = Path(path)
    data = json.loads(path.read_text())
    mesh = load_mesh(path.parent / data["mesh"])
    material = Material(data["material"]["young"], data["material"]["poisson"])
    boundary = build_boundary(mesh, data.get("constraints", []), data.get("loads", []))
    return mesh, material, boundary
