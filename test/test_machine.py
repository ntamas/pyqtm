from pyqtm.model import QuantumTuringMachine

from pytest import fixture, raises


@fixture
def machine() -> QuantumTuringMachine:
    machine = QuantumTuringMachine(num_states=5, num_symbols=3, tape_length=4)
    return machine


def test_machine_creation(machine: QuantumTuringMachine):
    assert machine.num_states == 5
    assert machine.num_symbols == 3
    assert machine.tape_length == 4

    assert list(machine.get_internal_state().get_probabilities()) == [1, 0, 0, 0, 0]
    assert list(machine.get_head_position().get_probabilities()) == [1, 0, 0, 0]

    num_tape_symbol_combinations = machine.num_symbols**machine.tape_length
    expected = [1.0] + [0.0] * (num_tape_symbol_combinations - 1)
    assert list(machine.get_tape_contents().get_probabilities()) == expected


def test_invalid_machine_creation():
    with raises(ValueError, match="at least one cell"):
        QuantumTuringMachine(num_states=5, num_symbols=3, tape_length=0)
    with raises(ValueError, match="at least two possible symbols"):
        QuantumTuringMachine(num_states=5, num_symbols=1, tape_length=4)
    with raises(ValueError, match="at least one internal state"):
        QuantumTuringMachine(num_states=0, num_symbols=3, tape_length=4)
