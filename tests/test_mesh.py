import meshio
import numpy as np
from numpy.testing import assert_allclose

from fem.input import box_mesh, load_mesh, surface_faces


def test_box_surface_is_outward_and_internal_faces_removed():
    mesh = box_mesh(size=(2, 3, 4), divisions=(2, 3, 4))
    faces, owners = surface_faces(mesh)
    assert len(faces) == 4 * (2 * 3 + 3 * 4 + 2 * 4)
    xyz = mesh.points[faces]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    to_interior = mesh.points[mesh.tetra[owners]].mean(axis=1) - xyz.mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", normals, to_interior) < 0)
    volume = (
        np.abs(np.linalg.det(mesh.points[mesh.tetra[:, 1:]] - mesh.points[mesh.tetra[:, :1]])).sum()
        / 6
    )
    assert_allclose(volume, 24)


def test_gmsh_physical_surface_group_is_preserved(tmp_path, tetra):
    path = tmp_path / "physical.msh"
    meshio.write(
        path,
        meshio.Mesh(
            tetra.points,
            [("triangle", np.array([[0, 1, 2]])), ("tetra", tetra.tetra)],
            cell_data={
                "gmsh:physical": [np.array([7]), np.array([8])],
                "gmsh:geometrical": [np.array([1]), np.array([1])],
            },
            field_data={"support": np.array([7, 2]), "solid": np.array([8, 3])},
        ),
        file_format="gmsh22",
        binary=False,
    )
    mesh = load_mesh(path)
    assert_allclose(mesh.groups["support"], [0, 1, 2])
    assert_allclose(mesh.groups["solid"], [0, 1, 2, 3])
