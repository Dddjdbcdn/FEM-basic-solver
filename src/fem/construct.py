"""Build a conforming Tet4 mesh from named, axis-aligned boxes in JSON."""

import argparse
import ast
import json
import operator
from itertools import product
from pathlib import Path

import meshio
import numpy as np

from .input import Mesh, surface_faces


def evaluate(value, parameters):
    """Allow numbers, parameter names, and arithmetic; no Python calls or attributes."""
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def read(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name):
            return parameters[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](read(node.left), read(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -read(node.operand)
        raise ValueError("Use numbers, parameter names, and +, -, *, / in expressions.")

    return read(ast.parse(str(value), mode="eval").body)


def read_boxes(data):
    if data.get("units", "m") != "m":
        raise ValueError("Use metres (units: m) for this format.")
    if not data["parts"]:
        raise ValueError("Add at least one box to parts.")
    parameters = {}
    for name, value in data.get("parameters", {}).items():
        parameters[name] = evaluate(value, parameters)

    names, lower, upper = [], [], []
    for part in data["parts"]:
        if set(part) - {"name", "origin", "size"}:
            raise ValueError("Parts support name, origin, and size only (axis-aligned boxes).")
        name = part["name"]
        origin = np.array([evaluate(v, parameters) for v in part["origin"]])
        size = np.array([evaluate(v, parameters) for v in part["size"]])
        if name in names or origin.shape != (3,) or size.shape != (3,):
            raise ValueError("Every part needs a unique name and three origin/size values.")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError(f"Part {name!r} needs finite coordinates and positive dimensions.")
        names.append(name)
        lower.append(origin)
        upper.append(origin + size)

    lower, upper = np.array(lower), np.array(upper)
    tolerance = np.max(upper.max(axis=0) - lower.min(axis=0)) * 1e-10
    if np.any(upper - lower <= tolerance):
        raise ValueError("A box dimension is too small relative to the object size.")
    neighbours = [[] for name in names]
    for a in range(len(names)):
        for b in range(a):
            overlap = np.minimum(upper[a], upper[b]) - np.maximum(lower[a], lower[b])
            if np.all(overlap > tolerance):
                raise ValueError(f"Parts {names[a]!r} and {names[b]!r} overlap. Use face joints.")
            if np.all(overlap >= -tolerance):
                if np.count_nonzero(overlap > tolerance) != 2:
                    raise ValueError(
                        f"Parts {names[a]!r} and {names[b]!r} only meet at an edge or point."
                    )
                neighbours[a].append(b)
                neighbours[b].append(a)

    connected = {0}
    pending = [0]
    while pending:
        for neighbour in neighbours[pending.pop()]:
            if neighbour not in connected:
                connected.add(neighbour)
                pending.append(neighbour)
    if len(connected) != len(names):
        raise ValueError("All parts must connect through faces into one bonded object.")

    mesh_size = evaluate(data["mesh_size"], parameters)
    if not np.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError("mesh_size must be positive and finite.")
    return names, lower, upper, mesh_size, tolerance


def build_mesh(data):
    names, lower, upper, mesh_size, tolerance = read_boxes(data)

    # Put every part boundary on the same grid, including partial face contacts.
    axes = []
    for axis in range(3):
        cuts = []
        for value in sorted(np.concatenate([lower[:, axis], upper[:, axis]])):
            if not cuts or value - cuts[-1] > tolerance:
                cuts.append(value)
        coordinates = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            count = max(1, int(np.ceil((end - start) / mesh_size - 1e-10)))
            coordinates.extend(np.linspace(start, end, count + 1)[:-1])
        axes.append(np.array(coordinates + [cuts[-1]]))

    centres = [(axis[:-1] + axis[1:]) / 2 for axis in axes]
    points, tetra, element_parts = [], [], []
    node_ids = {}

    def node_id(index):
        key = tuple(index)
        if key not in node_ids:
            node_ids[key] = len(points)
            points.append([axes[axis][index[axis]] for axis in range(3)])
        return node_ids[key]

    # Eight box corners, then six positively oriented tetrahedra around diagonal 0--7.
    offsets = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    patterns = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    for part in range(len(names)):
        indices = []
        for axis in range(3):
            indices.append(
                np.flatnonzero(
                    (centres[axis] >= lower[part, axis]) & (centres[axis] <= upper[part, axis])
                )
            )
        for index in product(*indices):
            corners = [node_id(np.array(index) + offset) for offset in offsets]
            for pattern in patterns:
                tetra.append([corners[corner] for corner in pattern])
                element_parts.append(part)

    mesh = Mesh(np.array(points), np.array(tetra, dtype=np.int32))
    element_parts = np.array(element_parts)
    for part, name in enumerate(names):
        mesh.parts[name] = np.flatnonzero(element_parts == part)
        mesh.groups[name] = np.unique(mesh.tetra[mesh.parts[name]])

    # Face groups select exterior triangles of named parts, not guessed CAD tags.
    faces, owners = surface_faces(mesh)
    face_groups = {}
    face_names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    for name, selection in data.get("groups", {}).items():
        if name in mesh.groups:
            raise ValueError(f"Group {name!r} duplicates a part name.")
        face = selection["face"]
        if face not in face_names:
            raise ValueError(f"Face must be one of {face_names}.")
        axis = "xyz".index(face[0])
        selected = np.zeros(len(faces), dtype=bool)
        for part_name in selection["parts"]:
            part = names.index(part_name)
            coordinate = lower[part, axis] if face.endswith("min") else upper[part, axis]
            on_plane = np.all(np.abs(mesh.points[faces, axis] - coordinate) <= tolerance, axis=1)
            selected |= (element_parts[owners] == part) & on_plane
        if not np.any(selected):
            raise ValueError(f"Group {name!r} matches no exterior faces.")
        face_groups[name] = faces[selected]
        mesh.groups[name] = np.unique(faces[selected])

    return mesh, element_parts, face_groups


def construct(path, output=None):
    path = Path(path)
    data = json.loads(path.read_text())
    mesh, element_parts, face_groups = build_mesh(data)
    output = Path(output) if output else path.with_suffix(".msh")
    output.parent.mkdir(parents=True, exist_ok=True)

    cells = [("tetra", mesh.tetra)]
    tags = [element_parts + 1]
    fields = {}
    for tag, name in enumerate(mesh.parts, start=1):
        fields[name] = np.array([tag, 3])
    for tag, (name, faces) in enumerate(face_groups.items(), start=len(mesh.parts) + 1):
        cells.append(("triangle", faces))
        tags.append(np.full(len(faces), tag))
        fields[name] = np.array([tag, 2])
    meshio.write(
        output,
        meshio.Mesh(
            mesh.points,
            cells,
            field_data=fields,
            cell_data={"gmsh:physical": tags, "gmsh:geometrical": tags},
        ),
        file_format="gmsh22",
        binary=False,
    )
    print(f"Mesh: {output} ({len(mesh.points):,} nodes, {len(mesh.tetra):,} tetrahedra)")

    if "material" in data:
        case = {
            "mesh": output.name,
            "material": data["material"],
            "constraints": data.get("constraints", []),
            "loads": data.get("loads", []),
        }
        case_path = output.with_suffix(".case.json")
        case_path.write_text(json.dumps(case, indent=2) + "\n")
        print(f"Case: {case_path}")
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct a bonded box model from JSON")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    mesh = construct(args.model, args.output)
    if args.preview:
        from .viewer import view

        view(mesh, title=args.model.stem, up="y")
