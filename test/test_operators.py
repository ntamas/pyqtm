from pyqtm.constants import SQRT_2_INV
from pyqtm.model import QuantumState
from pyqtm.model.operators import HADAMARD
from pytest import approx


def test_hadamard_twice():
    state = QuantumState.from_amplitudes([1, 0])

    state.apply(HADAMARD)
    assert list(state) == approx([SQRT_2_INV, SQRT_2_INV])

    state.apply(HADAMARD)
    assert list(state) == approx([1, 0])

    state = QuantumState.from_amplitudes([0, 1])
    state.apply(HADAMARD)
    assert list(state) == approx([SQRT_2_INV, -SQRT_2_INV])

    state.apply(HADAMARD)
    assert list(state) == approx([0, 1])
