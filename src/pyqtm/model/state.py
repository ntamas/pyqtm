from itertools import product
from typing import (
    cast,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from numpy import abs, array, cdouble, copy, double, dot, int32, matmul, outer, zeros
from numpy.linalg import norm
from numpy.typing import ArrayLike, NDArray

from .types import Amplitude, Operator
from .variable import DiscreteQuantumVariable

__all__ = ("QuantumState",)


C = TypeVar("C", bound="QuantumState")


def _calculate_index_multipliers(
    num_states_per_var: List[int],
) -> Tuple[NDArray[int32], int]:
    multipliers = zeros(len(num_states_per_var), dtype=int32)
    mul = 1
    for idx, x in enumerate(reversed(num_states_per_var), 1):
        multipliers[-idx] = mul
        mul *= x
    return multipliers, mul


class QuantumState(Sequence[Amplitude]):
    """Class representing one or more discrete quantum variables, each having
    a finite number of possible states, and an associated matrix of probability
    amplitudes representing the joint distribution of states.
    """

    _variables: List[DiscreteQuantumVariable]
    """The list of variables in this state object."""

    _variable_indices_by_name: Dict[str, int]
    """Dictionary mapping the names of named variables in this state object
    to the indices where they appear in the state object.
    """

    _amplitudes: NDArray[Amplitude]
    """A NumPy array containing the probability amplitudes of the joint
    distribution as a column vector.

    Note that even if you have multiple variables, the amplitudes are always
    stored as a single column vector as NumPy arrays are limited to 32
    dimensions. Use the `state_to_index()` method to determine the row index
    of a given state combination.
    """

    _index_multipliers: NDArray[int32]
    """Multipliers used when converting a tuple of individual state indices to
    the flat index of the probability amplitude vector.
    """

    @classmethod
    def from_amplitudes(cls, amplitudes: ArrayLike, *, name: Optional[str] = None):
        """Factory method that creates a quantum state object from a single
        variable with the given probability amplitudes.
        """
        amplitude_array = array(amplitudes, dtype=cdouble)

        # Ensure that the amplitude array is a column vector
        if (
            sum(1 for x in amplitude_array.shape if x == 1)
            >= len(amplitude_array.shape) - 1
        ):
            amplitude_array.shape = (amplitude_array.shape[0],)
        else:
            raise ValueError("input array must be one-dimensional")

        num_states = amplitude_array.shape[0]
        return cls(
            variables=[DiscreteQuantumVariable(name=name, num_states=num_states)],
            amplitudes=amplitude_array,
        )

    @classmethod
    def from_discrete(
        cls, num_states: int, initial_state: int = 0, *, name: Optional[str] = None
    ):
        """Factory method that creates a quantum state object from a classic
        discrete variable that has a given number of discrete states and a
        single initial state.
        """
        amplitudes = [0] * num_states
        amplitudes[initial_state] = 1
        return cls.from_amplitudes(amplitudes, name=name)

    def __init__(
        self,
        variables: Optional[Iterable[DiscreteQuantumVariable]] = None,
        amplitudes: Optional[ArrayLike] = None,
    ):
        """Creates an empty state object with no variables."""
        self._variables = list(variables) if variables is not None else []
        self._amplitudes = array(amplitudes if amplitudes is not None else [], dtype=cdouble)  # type: ignore
        if amplitudes is None:
            self._amplitudes.shape = (0,)
        self._amplitudes /= norm(self._amplitudes)
        self._notify_shape_changed()

    @property
    def variables(self) -> Sequence[DiscreteQuantumVariable]:
        """The list of variables in the state object.

        Do not modify this list directly; use `merge_from()` to add new
        variables.
        """
        return self._variables

    def apply(self: C, operator: Operator) -> C:
        """Applies the given operator (represented as a 2D NumPy array) to
        the state object, updating the probability amplitudes in place.

        The operator must be a unitary transformation; this function does not
        check whether this holds or not.

        Numeric inaccuracies may accumulate in this state object after applying
        an operator to it multiple times; in this case, call `normalize()`
        explicitly to restore the invariant that the sum of squares of the
        probability amplitudes must add up to 1.
        """
        matmul(operator, self._amplitudes, out=self._amplitudes)
        return self

    def get_probabilities(self) -> NDArray[double]:
        """Returns the absolute probabilities of observing each of the possible
        state combinations, in a NumPy column vector.
        """
        return abs(self._amplitudes) ** 2  # type: ignore

    def get_variable_by_name(self, name: str) -> DiscreteQuantumVariable:
        """Returns the variable corresponding to the given name in this state
        object.
        """
        return self._variables[self._variable_indices_by_name[name]]

    def iter_states(
        self, sort: bool = False, reverse: bool = False
    ) -> Iterable[Tuple[Tuple[int, ...], Amplitude]]:
        """Iterates over the state combinations and yields tuples consisting
        of the individual states of the variables and the probability amplitude
        corresponding to the given state combination.

        Args:
            sort: whether to sort the states first by their probabilities
            reverse: specifies whether to sort in ascending (``False``) or
                descending (``True``) order. Setting this to ``True`` implies
                setting `sort` to ``True``
        """
        if reverse:
            sort = True

        if not sort:
            states = [range(var.num_states) for var in self._variables]
            yield from zip(product(*states), self._amplitudes)
        else:
            indices = self.sorted_indices(reverse=reverse)
            items = list(self.iter_states())
            for index in indices:
                yield items[index]

    def merge_from(self: C, *others: C) -> C:
        """Merges the variables of another quantum state object into this
        object.
        """
        for other in others:
            self._variables.extend(other._variables)
            self._amplitudes = outer(self._amplitudes, other._amplitudes).reshape(-1)
        self._notify_shape_changed()
        return self

    def sorted_indices(self, reverse: bool = False) -> NDArray[int32]:
        """Returns a NumPy array of state indices, ordered from lowest
        probability to highest (or reversed).

        Args:
            reverse: whether to sort state indices with higher probabilities
                first (``True``) or last (``False``)
        """
        # We use a stable sort to preserve the natural ordering of states
        result = abs(self._amplitudes).argsort(kind="stable")
        return result[::-1] if reverse else result

    def state_to_index(self, indices: Sequence[int]) -> int:
        """Converts a tuple containing state indices of the individual variables
        to a combined index that can be used to look up the corresponding
        probability amplitude.
        """
        if not indices:
            return 0
        elif len(indices) == 1:
            return indices[0]
        else:
            return dot(indices, self._index_multipliers)

    def substate(self: C, indices: Union[int, str, Iterable[Union[int, str]]]) -> C:
        """Returns a substate consisting of a variable with a single index,
        multiple indices or a variable by name.
        """
        if not hasattr(indices, "__iter__") or isinstance(indices, str):
            return self.substate((indices,))  # type: ignore
        else:
            index_list = self._resolve_variable_names(indices)  # type: ignore

        if not index_list:
            return self.__class__()

        if index_list == list(range(len(self._variables))):
            # Special case, copying self
            variables = list(self._variables)
            amplitudes = copy(self._amplitudes)
        else:
            # True subset
            variables = [self._variables[i] for i in index_list]
            num_states_per_var = [var.num_states for var in variables]
            index_multipliers, total_reduced_states = _calculate_index_multipliers(
                num_states_per_var
            )

            amplitudes = zeros((total_reduced_states,), dtype=cdouble)

            # For the amplitudes, we need to sum them up approriately
            for state, amplitude in self.iter_states():
                reduced_state = [state[i] for i in index_list]
                reduced_index = dot(reduced_state, index_multipliers)
                amplitudes[reduced_index] += amplitude

        return self.__class__(variables=variables, amplitudes=amplitudes)

    def __getitem__(self, index: Union[int, Sequence[int]]) -> cdouble:
        """The probability amplitude at the given state index combination."""
        if isinstance(index, tuple):
            index = self.state_to_index(index)
        return self._amplitudes[index]  # type: ignore

    def __len__(self) -> int:
        """The total number of possible state combinations, i.e. the number of
        rows in the probability amplitude column vector.
        """
        return len(self._amplitudes)

    def _notify_shape_changed(self) -> None:
        """Handler that is called when the shape of the state object (i.e. the
        number of variables or the number of states for a variable) changed.
        """
        self._index_multipliers, _ = _calculate_index_multipliers(
            [var.num_states for var in self._variables]
        )
        self._variable_indices_by_name = {
            var.name: idx
            for idx, var in enumerate(self._variables)
            if var.name is not None
        }

    def _resolve_variable_names(
        self, names_or_indices: Iterable[Union[str, int]]
    ) -> List[int]:
        """Takes an iterable containing a mixture of variable indices and names,
        and returns a list that contains the same indices and where the names
        are replaced by the corresponding indices.
        """
        result: List[int] = []
        for name_or_index in names_or_indices:
            if isinstance(name_or_index, str):
                try:
                    result.append(self._variable_indices_by_name[name_or_index])
                except KeyError:
                    raise ValueError(f"no such variable: {name_or_index!r}")
            else:
                result.append(name_or_index)
        return result
