from numpy import cdouble
from numpy.typing import NDArray

__all__ = ("Amplitude", "Operator")


Amplitude = cdouble
"""Type alias for probability amplitudes."""

Operator = NDArray[cdouble]
"""Type alias for operators, i.e. 2D unitary matrices that can be applied on a
quantum state to obtain a new state.
"""
