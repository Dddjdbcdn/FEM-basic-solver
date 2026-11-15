import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .elasticity import element_stiffness, recover_stress
from .input import Result
from .results import average_stress, von_mises


def assemble_stiffness(mesh, element_k):
    size = 3 * len(mesh.points)
    K = sparse.lil_matrix((size, size), dtype=np.float64)

    for e, nodes in enumerate(mesh.tetra):
        for a in range(4):
            global_node_a = nodes[a]
            for i in range(3):  # Force direction
                local_row = 3 * a + i
                global_row = 3 * global_node_a + i

                for b in range(4):
                    global_node_b = nodes[b]
                    for j in range(3):  # Displacement direction
                        local_column = 3 * b + j
                        global_column = 3 * global_node_b + j

                        # Shared global pairs
                        K[global_row, global_column] += element_k[e, local_row, local_column]

    return K.tocsr()


def solve(mesh, material, boundary):
    element_k, gradients, volumes = element_stiffness(mesh, material)
    K = assemble_stiffness(mesh, element_k)

    force = boundary.force.ravel()
    fixed = boundary.fixed.ravel()
    free = ~fixed
    u = np.zeros(len(force))
    u[fixed] = boundary.values.ravel()[fixed]

    relative_residual = 0.0
    
    if np.any(free):
        K_ff = K[free][:, free]
        K_fc = K[free][:, fixed]
        rhs = force[free] - K_fc @ u[fixed]
        u[free] = spsolve(K_ff, rhs)
        if np.linalg.norm(rhs) > 0:
            relative_residual = np.linalg.norm(K_ff @ u[free] - rhs) / np.linalg.norm(rhs)

    # 4. Recover support forces, strain, and stress from the solved displacement.
    internal_force = K @ u
    reaction = np.zeros(len(force))
    reaction[fixed] = (internal_force - force)[fixed]
    displacement = u.reshape(-1, 3)
    strain, stress = recover_stress(mesh, material, displacement, gradients)
    nodal_stress = average_stress(mesh, stress, volumes)

    return Result(
        displacement=displacement,
        reaction=reaction.reshape(-1, 3),
        strain=strain,
        stress=stress,
        von_mises=von_mises(stress),
        nodal_stress=nodal_stress,
        nodal_von_mises=von_mises(nodal_stress),
        volumes=volumes,
        relative_residual=relative_residual,
        strain_energy=0.5 * u @ internal_force,
    )
