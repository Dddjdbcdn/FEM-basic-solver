# Parametric box model

A model is a list of named, axis-aligned boxes. Coordinates and lengths are in
metres. `origin` is the minimum x/y/z corner; `size` is the x/y/z length.
The supplied table uses y as the vertical direction.

The complete example is [mesh/table.json](../mesh/table.json). Its editor schema
is [mesh/box-model.schema.json](../mesh/box-model.schema.json).

## Parameters and parts

```json
{
  "$schema": "box-model.schema.json",
  "units": "m",
  "parameters": {
    "height": 0.8,
    "top_thickness": 0.08
  },
  "mesh_size": 0.04,
  "parts": [
    {
      "name": "tabletop",
      "origin": [0, "height - top_thickness", 0],
      "size": [1.2, "top_thickness", 0.8]
    },
    {
      "name": "leg",
      "origin": [0.04, 0, 0.04],
      "size": [0.08, "height - top_thickness", 0.08]
    }
  ]
}
```

That small example makes a top and one attached leg; the full table example
spells out all four legs. Change a parameter once and rebuild to move/resize all
parts that reference it. Parameters may reference earlier parameters.
Expressions accept numbers, names, parentheses, and `+`, `-`, `*`, `/`.
They are parsed as arithmetic; Python calls and attributes are not supported.

Every part needs a unique name and three finite origin/size components. Sizes
must be positive. Tiny dimensions indistinguishable at the model's coordinate
tolerance are rejected. Parts must form one object through face contacts;
volume overlaps, gaps between disconnected components, and edge/point-only
contacts are rejected before meshing. Rotated boxes and other shapes are not in
the format yet. The constructor accepts only `name`, `origin`, and `size` on parts.

## Named exterior faces

Add a `groups` object:

```json
"groups": {
  "feet": {"parts": ["leg"], "face": "ymin"},
  "load_surface": {"parts": ["tabletop"], "face": "ymax"}
}
```

Faces can be `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, or `zmax`.
The direction is relative to each named box: `ymin` means that part's lowest-y
face. A group can contain that face from several parts, as the full table's
`feet` group does. Only exposed exterior triangles are selected. A fully bonded
internal face is not a load boundary, and a group with no exterior faces is an
error. Group names must differ from part names.

The mesh preserves each part as a named tetrahedron set. Our loader exposes it
in `mesh.parts`, and exposes its node set in `mesh.groups`. Exterior face groups
also become node sets in `mesh.groups`, ready for the existing solver.

## Analysis in the same JSON

Optionally add:

```json
"material": {"young": 70000000000, "poisson": 0.3},
"constraints": [
  {"select": {"group": "feet"}, "displacement": [0, 0, 0]}
],
"loads": [
  {"select": {"group": "load_surface"}, "force": [0, -1000, 0], "distribution": "total"}
]
```

With `material` present, the constructor writes an analysis `.case.json` alongside
the mesh. It references the named groups. Force is in newtons, displacement in
metres, and Young's modulus in pascals. These are nodal forces; `total` splits the
vector equally among the selected nodes. `null` displacement leaves that
component free. One material applies to all parts. Shared nodes at joints mean
perfectly bonded displacement continuity, not hinges or contact.

## Build, preview, solve

```bash
python -m fem.construct mesh/table.json
python -m fem.construct mesh/table.json --preview
python -m fem mesh/table.msh --mesh-only --up y
python -m fem mesh/table.case.json --up y
```

The source JSON remains the editable definition. `table.msh` and `table.case.json`
are generated files and are overwritten on rebuilding. Use `--output output/demo.msh`
to choose another destination; its generated case is `output/demo.case.json`.
Without `material`, only the mesh is written.

`mesh_size` is the maximum grid interval. Every part boundary adds a grid plane,
and those intervals are subdivided to meet the requested size. The same grid and
six-tetrahedron split are used on both sides of each joint, so interface faces
match without hanging nodes. This can refine other parts as well as the part
whose dimensions you changed. Smaller mesh sizes increase solve time.
