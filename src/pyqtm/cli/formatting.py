from math import degrees
from numpy import abs, angle
from typing import List, Sequence

from pyqtm.model.types import Amplitude

__all__ = ("format_machine_state",)


def format_amplitude(amplitude: Amplitude) -> str:
    """Formats a probability amplitude using a human-readable magnitude-and-phase
    representation.
    """
    r, phi = (abs(amplitude) ** 2), degrees(angle(amplitude))
    return f"{r:.3f} @ {phi:.1f}\u00b0"


def format_machine_state(state: Sequence[int]) -> str:
    """Returns a nice formatted representation of the state of a quantum Turing
    machine that is suitable for printing on the console.
    """
    parts: List[str] = []

    parts.append(f"State: {state[0]} ")
    parts.append("  Tape: ")

    head_index = state[1]

    for index, symbol in enumerate(state[2:]):
        formatted_symbol = "_" if symbol == 0 else str(symbol - 1)
        if index == head_index:
            formatted_symbol = f"[{formatted_symbol}]"
        elif index == head_index - 1:
            pass
        else:
            formatted_symbol = formatted_symbol + " "
        parts.append(formatted_symbol)

    return "".join(parts)
