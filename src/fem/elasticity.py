import numpy as np
import taichi as ti


def init_runtime():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f32)


@ti.kernel
def _geometry(
    points: ti.types.ndarray(dtype=ti.f64, ndim=2),
    cells: ti.types.ndarray(dtype=ti.i32, ndim=2),
    gradients: ti.types.ndarray(dtype=ti.f64, ndim=3),
    volumes: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    # dN_a/d(r,s,t) for N = [1-r-s-t, r, s, t].
    reference_gradients = ti.Matrix(
        [
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dt=ti.f64,
    )
    for e in range(cells.shape[0]):
        x = ti.Matrix.zero(ti.f64, 4, 3)
        for a, i in ti.static(ti.ndrange(4, 3)):
            x[a, i] = points[cells[e, a], i]
        jacobian = ti.Matrix.cols([x[1, :] - x[0, :], x[2, :] - x[0, :], x[3, :] - x[0, :]])
        inverse = jacobian.inverse()
        volumes[e] = ti.abs(jacobian.determinant()) / 6.0

        # G_x = G_reference @ J_inverse.
        physical_gradients = reference_gradients @ inverse
        for a, i in ti.static(ti.ndrange(4, 3)):
            gradients[e, a, i] = physical_gradients[a, i]


@ti.kernel
def _stiffness(
    gradients: ti.types.ndarray(dtype=ti.f64, ndim=3),
    volumes: ti.types.ndarray(dtype=ti.f64, ndim=1),
    lam: ti.f64,
    mu: ti.f64,
    stiffness: ti.types.ndarray(dtype=ti.f64, ndim=3),
):
    # stress_vector = D @ strain_engineering.
    D = ti.Matrix(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0], # sigma_xx
            [lam, lam + 2 * mu, lam, 0, 0, 0], # sigma_yy
            [lam, lam, lam + 2 * mu, 0, 0, 0], # sigma_zz
            [0, 0, 0, mu, 0, 0], # tau_xy
            [0, 0, 0, 0, mu, 0], # tau_yz
            [0, 0, 0, 0, 0, mu], # tau_xz
        ],
        dt=ti.f64,
    )
    for e in range(volumes.shape[0]):
        g0x, g0y, g0z = gradients[e, 0, 0], gradients[e, 0, 1], gradients[e, 0, 2]
        g1x, g1y, g1z = gradients[e, 1, 0], gradients[e, 1, 1], gradients[e, 1, 2]
        g2x, g2y, g2z = gradients[e, 2, 0], gradients[e, 2, 1], gradients[e, 2, 2]
        g3x, g3y, g3z = gradients[e, 3, 0], gradients[e, 3, 1], gradients[e, 3, 2]

        # strain_engineering = B @ nodal_displacements.
        B = ti.Matrix(
            [
                [g0x, 0, 0, g1x, 0, 0, g2x, 0, 0, g3x, 0, 0],  # epsilon_xx
                [0, g0y, 0, 0, g1y, 0, 0, g2y, 0, 0, g3y, 0],  # epsilon_yy
                [0, 0, g0z, 0, 0, g1z, 0, 0, g2z, 0, 0, g3z],  # epsilon_zz
                [g0y, g0x, 0, g1y, g1x, 0, g2y, g2x, 0, g3y, g3x, 0],  # gamma_xy
                [0, g0z, g0y, 0, g1z, g1y, 0, g2z, g2y, 0, g3z, g3y],  # gamma_yz
                [g0z, 0, g0x, g1z, 0, g1x, g2z, 0, g2x, g3z, 0, g3x],  # gamma_xz
            ],
            dt=ti.f64,
        )

        # B and D are constant inside Tet4
        K_e = volumes[e] * (B.transpose() @ D @ B)
        for row, column in ti.ndrange(12, 12):
            stiffness[e, row, column] = K_e[row, column]


@ti.kernel
def _recover(
    cells: ti.types.ndarray(dtype=ti.i32, ndim=2),
    gradients: ti.types.ndarray(dtype=ti.f64, ndim=3),
    displacement: ti.types.ndarray(dtype=ti.f64, ndim=2),
    lam: ti.f64,
    mu: ti.f64,
    strain: ti.types.ndarray(dtype=ti.f64, ndim=2),
    stress: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    for e in range(cells.shape[0]):
        grad_u = ti.Matrix.zero(ti.f64, 3, 3)
        for a in range(4):
            for i, j in ti.static(ti.ndrange(3, 3)):
                grad_u[i, j] += displacement[cells[e, a], i] * gradients[e, a, j]
        eps = (grad_u + grad_u.transpose()) * 0.5
        sigma = 2.0 * mu * eps + lam * eps.trace() * ti.Matrix.identity(ti.f64, 3)
        # Store the six independent components in order: xx, yy, zz, xy, yz, xz.
        strain[e, 0] = eps[0, 0]
        strain[e, 1] = eps[1, 1]
        strain[e, 2] = eps[2, 2]
        strain[e, 3] = eps[0, 1]
        strain[e, 4] = eps[1, 2]
        strain[e, 5] = eps[0, 2]
        stress[e, 0] = sigma[0, 0]
        stress[e, 1] = sigma[1, 1]
        stress[e, 2] = sigma[2, 2]
        stress[e, 3] = sigma[0, 1]
        stress[e, 4] = sigma[1, 2]
        stress[e, 5] = sigma[0, 2]


def geometry(mesh):
    init_runtime()
    gradients = np.empty((len(mesh.tetra), 4, 3), dtype=np.float64)
    volumes = np.empty(len(mesh.tetra), dtype=np.float64)
    _geometry(mesh.points, mesh.tetra, gradients, volumes)
    return gradients, volumes


def element_stiffness(mesh, material):
    gradients, volumes = geometry(mesh)
    stiffness = np.empty((len(mesh.tetra), 12, 12), dtype=np.float64)
    lam, mu = material.lame
    _stiffness(gradients, volumes, lam, mu, stiffness)
    return stiffness, gradients, volumes


def recover_stress(mesh, material, displacement, gradients):
    init_runtime()
    strain = np.empty((len(mesh.tetra), 6), dtype=np.float64)
    stress = np.empty_like(strain)
    lam, mu = material.lame
    _recover(mesh.tetra, gradients, displacement, lam, mu, strain, stress)
    return strain, stress
