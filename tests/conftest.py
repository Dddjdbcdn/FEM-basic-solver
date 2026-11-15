import numpy as np
import pytest

from fem.input import Material, Mesh


@pytest.fixture
def tetra():
    # Deliberately skewed and translated, so reference-element shortcuts fail.
    return Mesh(
        np.array([[0.2, -0.1, 0.3], [1.3, 0.1, 0.4], [0.4, 1.2, 0.2], [0.1, 0.3, 1.5]]),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def material():
    return Material(70e9, 0.3)
