from dataclasses import dataclass
from typing import Optional
from venv import create

__all__ = ("DiscreteQuantumVariable", "create_qubit")


@dataclass(frozen=True)
class DiscreteQuantumVariable:
    """Metadata about a single discrete quantum variable, including its name
    (if applicable) and the number of possible states that the variable can
    be in.
    """

    num_states: int
    """The number of states of the variable."""

    name: Optional[str] = None
    """The name of the variable."""


def create_qubit(name: str, *, num_states: int = 2) -> DiscreteQuantumVariable:
    """Creates a qubit.

    Args:
        num_states: the number of disjoint base states of the qubit
    """
    return DiscreteQuantumVariable(name=name, num_states=num_states)
