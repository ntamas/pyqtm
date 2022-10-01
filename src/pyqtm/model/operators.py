"""Factory functions for unitary matrices that can be applied on a quantum
state object of appropriate size.
"""

from numpy import array, eye

from pyqtm.constants import SQRT_2_INV

from .state import QuantumState
from .types import Amplitude, Operator

__all__ = ("HADAMARD",)

HADAMARD = array([[SQRT_2_INV, SQRT_2_INV], [SQRT_2_INV, -SQRT_2_INV]])


def create_identity_operator_for(state: QuantumState) -> Operator:
    """Creates an identity operator that operates on the given quantum state
    (or any other quantum state with the same number of associated probability
    amplitudes).
    """
    return eye(len(state), dtype=Amplitude)
