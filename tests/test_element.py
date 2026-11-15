import numpy as np
from numpy.testing import assert_allclose

from fem.elasticity import element_stiffness, recover_stress
from fem.results import von_mises


def test_stiffness_against_independent_b_matrix(tetra, material):
    stiffness, gradients, volumes = element_stiffness(tetra, material)
    # Solve the affine shape-function coefficients independently in NumPy.
    coefficients = np.linalg.inv(np.column_stack([np.ones(4), tetra.points]))
    grad = coefficients[1:].T
    b = np.zeros((6, 12))
    for a, (x, y, z) in enumerate(grad):
        b[:, 3 * a : 3 * a + 3] = [[x, 0, 0], [0, y, 0], [0, 0, z], [y, x, 0], [0, z, y], [z, 0, x]]
    lam, mu = material.lame
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] += 2 * mu
    d[3:, 3:] = np.eye(3) * mu
    volume = abs(np.linalg.det(tetra.points[1:] - tetra.points[0])) / 6
    assert_allclose(gradients[0], grad, rtol=1e-13, atol=1e-14)
    assert_allclose(volumes[0], volume, rtol=1e-13)
    assert_allclose(stiffness[0], volume * b.T @ d @ b, rtol=1e-12, atol=1e-5)
    assert_allclose(stiffness[0], stiffness[0].T, atol=1e-5)
    eigenvalues = np.linalg.eigvalsh(stiffness[0])
    assert_allclose(eigenvalues[:6], 0, atol=eigenvalues[-1] * 1e-13)
    assert np.all(eigenvalues[6:] > eigenvalues[-1] * 1e-3)


def test_rigid_translation_and_rotation_have_zero_strain(tetra, material):
    _, gradients, _ = element_stiffness(tetra, material)
    rotation = np.array([[0, -0.002, 0.003], [0.002, 0, -0.001], [-0.003, 0.001, 0]])
    u = tetra.points @ rotation.T + np.array([0.01, -0.02, 0.03])
    strain, stress = recover_stress(tetra, material, u, gradients)
    assert_allclose(strain, 0, atol=1e-16)
    assert_allclose(stress, 0, atol=1e-5)


def test_affine_strain_and_shear_convention(tetra, material):
    _, gradients, _ = element_stiffness(tetra, material)
    a = np.array([[0.001, 0.002, -0.003], [0.004, -0.002, 0.001], [0.002, 0.003, 0.004]])
    eps = (a + a.T) / 2
    lam, mu = material.lame
    sigma = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
    components = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    strain, stress = recover_stress(tetra, material, tetra.points @ a.T, gradients)
    assert_allclose(strain[0], [eps[i, j] for i, j in components], atol=1e-16)
    assert_allclose(stress[0], [sigma[i, j] for i, j in components], rtol=1e-12)


def test_von_mises_known_states():
    assert_allclose(
        von_mises(
            np.array(
                [
                    [100, 100, 100, 0, 0, 0],
                    [100, 0, 0, 0, 0, 0],
                    [0, 0, 0, 10, 0, 0],
                ]
            )
        ),
        [0, 100, np.sqrt(3) * 10],
    )
