from numpy import allclose, cdouble, eye
from numpy.typing import NDArray

__all__ = ("is_unitary",)

DEBUG: bool = False


def is_unitary(m: NDArray[cdouble]) -> bool:
    """Returns whether the given NumPy matrix is unitary."""

    if DEBUG:
        from numpy import set_printoptions

        set_printoptions(linewidth=200)
        print(repr(m))
        print(repr(m.conj().T))
        print(m.dot(m.conj().T))

    return allclose(eye(m.shape[0]), m.dot(m.conj().T))
