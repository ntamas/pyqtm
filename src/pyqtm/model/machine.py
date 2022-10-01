from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from re import X
from numpy import cdouble, indices, int32
from numpy.typing import NDArray
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from .operators import create_identity_operator_for
from .state import QuantumState
from .types import Amplitude, Operator

__all__ = ("QuantumTuringMachine",)


def parse_amplitude_from_json(
    value: Union[str, Sequence[float], float, int]
) -> Amplitude:
    """Parses a probability amplitude from one of the various representations
    we use in the JSON formatted machine specification input.
    """
    if isinstance(value, (int, float)):
        return cdouble(value)
    elif isinstance(value, str):
        return cdouble(complex(value.replace(" ", "")))
    elif hasattr(value, "__getitem__"):
        if len(value) == 1:
            return parse_amplitude_from_json(value[0])
        elif len(value) == 2:
            return cdouble(float(value[0]) + 1j * float(value[1]))

    raise ValueError(f"cannot parse amplitude from: {value!r}")


def symbol_to_int(symbol: Union[str, int]) -> int:
    """Converts a string or an integer representing a symbol index to an
    integer.
    """
    if isinstance(symbol, str):
        index = 0 if symbol == "_" else (int(symbol) + 1)
    elif isinstance(symbol, int):
        index = symbol + 1
    else:
        raise ValueError("only integers and strings can be converted to symbols")
    if index < 0:
        raise ValueError("symbol indices must be non-negative")
    return index


class Direction(Enum):
    """Possible movement directions in a single rule of a quantum Turing machine."""

    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"

    @staticmethod
    def parse_from_json(value: Optional[Union[int, str, "Direction"]]) -> "Direction":
        """Parses a machine head direction from one of the various representations
        we use in the JSON formatted machine specification input.
        """
        if value is None:
            return Direction.STAY

        elif isinstance(value, int):
            if value >= -1 and value <= 1:
                return [Direction.LEFT, Direction.STAY, Direction.RIGHT][value + 1]

        elif isinstance(value, str):
            return Direction(value)

        elif isinstance(value, Direction):
            return value

        raise ValueError(f"cannot parse direction from: {value!r}")

    def apply_to(self, tape_position: int, tape_length: int) -> int:
        """Returns the new position of the tape head if a movement of this
        direction is applied.

        It is assumed that the tape is circular, so moving left from the first
        position will yield the last position and vice versa.

        Args:
            tape_position: the current tape position
            tape_length: the number of cells on the tape

        Returns:
            the new tape position
        """
        if self is Direction.RIGHT:
            offset = 1
        elif self is Direction.LEFT:
            offset = -1
        else:
            return tape_position
        return (tape_position + offset) % tape_length


@dataclass
class Rule:
    """A single rule in a quantum Turing machine."""

    amplitude: Amplitude
    """The probability amplitude corresponding to this rule."""

    current_state: int
    """The state to which this rule applies."""

    symbol: int
    """The symbol under the head of the Turing machine."""

    maybe_next_state: Optional[int] = None
    """The state to transition to when the given symbol is read in the given
    state; ``None`` if the state should stay the same.
    """

    maybe_symbol_to_write: Optional[int] = None
    """The symbol to write to the tape after the rule has been applied; ``None``
    if the original symbol should be kept.
    """

    direction: Direction = Direction.STAY
    """The direction to move the head in after the symbol has been written."""

    @classmethod
    def from_json(cls, obj):
        if not isinstance(obj, dict) and not isinstance(obj, list):
            raise ValueError("rule specification must be a list or a dictionary")

        data: Tuple[int, int, Optional[int], Optional[int], Direction, Amplitude]

        if isinstance(obj, dict):
            # Recode into a tuple
            data = (
                int(obj["current_state"]),
                obj["symbol"],
                int(obj["next_state"]) if obj.get("next_state") is not None else None,
                obj.get("symbol_to_write"),
                obj.get("direction", "stay"),
                1 + 0j,
            )
        else:
            data = tuple(obj)

        # Handle missing values at the end of the tuple
        if len(data) < 2:
            raise ValueError(
                "rule specification needs at least a current state and a symbol"
            )
        data = data[:1] + (symbol_to_int(data[1]),) + data[2:]  # type: ignore
        if len(data) < 3:
            data = data + (None,)  # type: ignore
        if len(data) < 4:
            data = data + (None,)  # type: ignore
        if data[3] is not None:
            data = data[:3] + (symbol_to_int(data[3]),) + data[4:]  # type: ignore
        if len(data) < 5:
            data = data + (Direction.STAY,)  # type: ignore
        data = data[:4] + (Direction.parse_from_json(data[4]),) + data[5:]  # type: ignore
        if len(data) < 6:
            data = data + (1,)  # type: ignore
        if len(data) != 6:
            raise ValueError(
                "rule specification contains too many items; at most 6 items are expected"
            )
        data = data[:5] + (parse_amplitude_from_json(data[5]),)  # type: ignore

        return cls(
            current_state=data[0],
            symbol=data[1],
            maybe_next_state=data[2],
            maybe_symbol_to_write=data[3],
            direction=data[4],
            amplitude=data[5],
        )

    @property
    def next_state(self) -> int:
        """The state to transition to after the rule has been applied."""
        return (
            self.maybe_next_state
            if self.maybe_next_state is not None
            else self.current_state
        )

    @property
    def symbol_to_write(self) -> int:
        """The symbol to write to the tape after the rule has been applied."""
        return (
            self.maybe_symbol_to_write
            if self.maybe_symbol_to_write is not None
            else self.symbol
        )

    @property
    def max_state_index(self) -> int:
        """Returns the maximum state index that is used in this rule."""
        return max(self.next_state, self.current_state)

    @property
    def max_symbol_index(self) -> int:
        """Returns the maximum symbol index that is used in this rule."""
        return max(self.symbol_to_write, self.symbol)


C = TypeVar("C", bound="QuantumTuringMachine")


class QuantumTuringMachine:
    """A quantum Turing machine with a single finite tape."""

    _state: QuantumState
    """The current state of the machine."""

    _operator: Operator
    """The operator representing the current ruleset of the machine."""

    _num_symbols: int
    """The number of symbols used on the tape of the machine."""

    @classmethod
    def from_json(cls, obj):
        if not isinstance(obj, dict):
            raise ValueError("machine specification must be a dictionary")

        rule_spec = obj.get("rules", [])
        if not isinstance(rule_spec, list):
            raise ValueError("rules in machine specification must be a list")

        rules = [
            Rule.from_json(item) for item in rule_spec if not isinstance(item, str)
        ]
        tape: Union[str, Sequence[int]] = obj.get("tape", "_")

        num_states = max((rule.max_state_index for rule in rules), default=0) + 1
        num_symbols = max((rule.max_symbol_index for rule in rules), default=0) + 1
        tape_length = len(tape)

        if num_symbols < 2:
            num_symbols = 2

        input_str_or_list = obj.get("tape", "")
        input: List[int] = []
        for symbol in input_str_or_list:
            symbol_index = symbol_to_int(symbol)
            if symbol_index < 0 or symbol_index >= num_symbols:
                raise ValueError(
                    f"symbol index {symbol_index} in input is out of range"
                )

            input.append(symbol_index)

        return (
            cls(num_states=num_states, num_symbols=num_symbols, tape_length=tape_length)
            .set_rules(rules)
            .set_tape_contents(input)
        )

    def __init__(self, num_states: int, num_symbols: int = 3, tape_length: int = 4):
        """Constructor.

        Args:
            num_states: number of internal states of the machine. The machine
                starts from state zero.
            num_symbols: number of symbols that can be used on the tape, _including_
                the blank symbol.
            tape_length: number of cells on the tape of the machine.
        """
        self._num_symbols = num_symbols
        self._state = self._create_initial_state(num_states, num_symbols, tape_length)
        self._operator = create_identity_operator_for(self._state)

    def _create_initial_state(
        self, num_states: int, num_symbols: int, tape_length: int
    ) -> QuantumState:
        """Creates the initial state object of the machine."""
        if tape_length < 1:
            raise ValueError("tape must have at least one cell")
        if num_symbols < 2:
            raise ValueError("tape must have at least two possible symbols per cell")
        if num_states < 1:
            raise ValueError("machine must have at least one internal state")

        internal_state = QuantumState.from_discrete(num_states, name="internal_state")
        head_position = QuantumState.from_discrete(tape_length, name="head_position")
        tape_cells = self._create_tape_cells(tape_length=tape_length)
        return internal_state.merge_from(head_position, *tape_cells)

    def get_head_position(self) -> QuantumState:
        """Returns the substate that represents the position of the head."""
        return self._state.substate("head_position")

    def get_internal_state(self) -> QuantumState:
        """Returns the substate that represents the internal state of the machine."""
        return self._state.substate("internal_state")

    def get_tape_contents(self) -> QuantumState:
        """Returns the substate that represents the tape cells of the machine."""
        return self._state.substate(range(2, 2 + self.tape_length))

    @property
    def num_states(self) -> int:
        """Returns the number of internal states of the machine."""
        return self._state.get_variable_by_name("internal_state").num_states

    @property
    def num_symbols(self) -> int:
        """Returns the number of possible symbols on the tape of the machine."""
        return self._num_symbols

    @property
    def state(self) -> QuantumState:
        """The entire state of the machine, including its internal state, the
        position of the tape head and the tape itself.
        """
        return self._state

    @property
    def tape_length(self) -> int:
        """Returns the number of cells on the tape of the machine."""
        return len(self._state.variables) - 2

    def set_rules(self: C, rules: Sequence[Rule]) -> C:
        """Sets the rules of the quantum Turing machine, recalculating the
        unitary operator behind the machine at the end.
        """
        # column i in self._operator contains the probability amplitudes of
        # transitioning from state i to each individual other state. The
        # column must be normalized such that \sum_j |m_{ij}|^2 = 1.
        #
        # Note that it is enough to normalize column-wise only once, at the
        # end when we have added all the rules.

        # First, we need to group the rules by (state, symbol) combinations
        rules_by_current_state_and_symbol: Dict[
            Tuple[int, int], List[Rule]
        ] = defaultdict(list)
        for rule in rules:
            key = rule.current_state, rule.symbol
            rules_by_current_state_and_symbol[key].append(rule)

        op = self._operator

        # Clear the operator matrix
        op *= 0

        # We iterate over each column of the operator. Each column represents
        # one possible full state of the quantum Turing machine. From the
        # column index, we can determine what the current internal state index
        # of that column is and what symbol the tape head points to. Then, we
        # can look up the matching list of rules and fill the cells in the
        # column.
        for index, (full_state, _) in enumerate(self._state.iter_states()):
            internal_state, tape_position = full_state[0], full_state[1]
            key = internal_state, full_state[tape_position + 2]
            rules_for_this_column = rules_by_current_state_and_symbol.get(key)
            if rules_for_this_column:
                # There is at least one rule for this column so fill the cells
                for rule in rules_for_this_column:
                    next_tape_position = rule.direction.apply_to(
                        tape_position, self.tape_length
                    )
                    new_tape_contents = list(full_state[2:])
                    new_tape_contents[tape_position] = rule.symbol_to_write
                    next_full_state = (
                        rule.next_state,
                        next_tape_position,
                        *new_tape_contents,
                    )
                    target_index = self._state.state_to_index(next_full_state)
                    op[target_index, index] += rule.amplitude
            else:
                # There are no rules for this column so just set the diagonal
                # cell to 1 + 0j
                op[index, index] = 1 + 0j

        # Normalize the operator now that all column sums are nonzeros
        self._operator = op / ((abs(op) ** 2).sum(axis=0) ** 0.5)

        return self

    def set_tape_contents(
        self: C, symbols: Sequence[int], *, tape_length: Optional[int] = None
    ) -> C:
        """Sets the contents of the tape of the machine to the given sequence
        of symbols.

        Args:
            symbols: the symbols to write on the tape, starting from index zero.
                If it is shorter than the length of the tape, the rest is
                filled with empty symbols.
            tape_length: the length of the tape; ``None`` means to use the
                current length

        Raises:
            ValueError: if the input sequence of symbols contains an invalid
                symbol index or if it is longer than the tape of the machine
        """
        if tape_length is None:
            tape_length = self.tape_length

        if len(symbols) > tape_length:
            raise ValueError(
                f"tape is only {tape_length} symbols long, got {len(symbols)}"
            )

        if any(symbol < 0 or symbol >= self.num_symbols for symbol in symbols):
            raise ValueError(f"input contains an invalid symbol")

        state_without_tape = self._state.substate(["internal_state", "head_position"])
        tape_cells = self._create_tape_cells(symbols, tape_length)

        self._state = state_without_tape.merge_from(*tape_cells)

        return self

    def simulate(self: C, steps: int = 1) -> C:
        """Simulates the quantum Turing machine for the given number of steps."""
        for _ in range(steps):
            self._state.apply(self._operator)
        return self

    def _create_tape_cells(
        self, symbols: Iterable[int] = (), tape_length: Optional[int] = None
    ) -> Sequence[QuantumState]:
        """Creates a list of state variables, one for each cell of the tape,
        initialized with the given list of symbols.
        """
        if tape_length is None:
            tape_length = self.tape_length

        result = [
            QuantumState.from_discrete(self.num_symbols, symbol, name=f"tape{index}")
            for index, symbol in enumerate(islice(symbols, tape_length))
        ]
        while len(result) < tape_length:
            index = len(result)
            result.append(
                QuantumState.from_discrete(self.num_symbols, name=f"tape{index}")
            )
        return result
