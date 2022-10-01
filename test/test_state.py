from math import sqrt
from pyqtm.model import create_qubit, QuantumState
from pytest import approx, fixture


@fixture
def qstate() -> QuantumState:
    a = QuantumState.from_amplitudes([1 / sqrt(2), 0, 1 / sqrt(2)], name="a")
    b = QuantumState.from_amplitudes([0, 1], name="b")
    c = QuantumState.from_amplitudes([1 / sqrt(2), 1 / sqrt(2)], name="c")
    return a.merge_from(b, c)


def test_state_single_bit():
    state = QuantumState.from_amplitudes([1, 0])

    assert len(state.variables) == 1
    assert state.variables[0].num_states == 2

    # Test iteration, sequence protocol
    assert list(state) == [1 + 0j, 0 + 0j]

    # Test indexing
    assert state[0] == 1 + 0j
    assert state[1] == 0 + 0j


def test_state_two_bits():
    a = QuantumState.from_amplitudes([1 / sqrt(2), 0, 1 / sqrt(2)])
    b = QuantumState.from_amplitudes([0, 1])

    state = a.merge_from(b)
    assert state is a

    assert len(state.variables) == 2
    assert state.variables[0].num_states == 3
    assert state.variables[1].num_states == 2

    # Test iteration, sequence protocol
    assert list(state) == approx(
        [
            0 + 0j,
            1 / sqrt(2) + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            1 / sqrt(2) + 0j,
        ]
    )

    # Test indexing
    assert state[0] == 0 + 0j
    assert state[1] == approx(1 / sqrt(2) + 0j)
    assert state[2] == 0 + 0j
    assert state[3] == 0 + 0j
    assert state[4] == 0 + 0j
    assert state[5] == approx(1 / sqrt(2) + 0j)
    assert state[0, 0] == 0 + 0j
    assert state[0, 1] == approx(1 / sqrt(2) + 0j)
    assert state[1, 0] == 0 + 0j
    assert state[1, 1] == 0 + 0j
    assert state[2, 0] == 0 + 0j
    assert state[2, 1] == approx(1 / sqrt(2) + 0j)


def test_state_three_bits(qstate):
    state = qstate

    assert len(state.variables) == 3
    assert state.variables[0].num_states == 3
    assert state.variables[1].num_states == 2
    assert state.variables[2].num_states == 2

    # Test iteration, sequence protocol
    expected = [
        0 + 0j,
        0 + 0j,
        1 / 2 + 0j,
        1 / 2 + 0j,
        0,
        0,
        0,
        0,
        0,
        0,
        1 / 2 + 0j,
        1 / 2 + 0j,
    ]
    assert list(state) == approx(expected)

    # Test indexing
    for i in range(len(state)):
        assert state[i] == approx(expected[i])

    assert state[0, 0, 0] == expected[0]
    assert state[0, 0, 1] == expected[1]
    assert state[0, 1, 0] == approx(expected[2])
    assert state[0, 1, 1] == approx(expected[3])
    assert state[1, 0, 0] == expected[4]
    assert state[1, 0, 1] == expected[5]
    assert state[1, 1, 0] == expected[6]
    assert state[1, 1, 1] == expected[7]
    assert state[2, 0, 0] == expected[8]
    assert state[2, 0, 1] == expected[9]
    assert state[2, 1, 0] == approx(expected[10])
    assert state[2, 1, 1] == approx(expected[11])


def test_get_probabilities(qstate):
    expected = [0, 0, 1 / 4, 1 / 4, 0, 0, 0, 0, 0, 0, 1 / 4, 1 / 4]
    assert qstate.get_probabilities() == approx(expected)

    state = QuantumState.from_amplitudes([1, 1])
    assert list(state.get_probabilities()) == approx([0.5, 0.5])


def test_substate_empty(qstate):
    state = qstate.substate(())
    assert len(state.variables) == 0
    assert list(state.get_probabilities()) == []


def test_substate_single_var(qstate):
    state = qstate.substate(0)

    assert len(state.variables) == 1
    assert state.variables[0].num_states == 3
    assert state[0] == approx(1 / sqrt(2))
    assert state[1] == 0
    assert state[2] == approx(1 / sqrt(2))

    state = qstate.substate(1)

    assert len(state.variables) == 1
    assert state.variables[0].num_states == 2
    assert state[0] == 0
    assert state[1] == 1

    state = qstate.substate(2)

    assert len(state.variables) == 1
    assert state.variables[0].num_states == 2
    assert state[0] == approx(1 / sqrt(2))
    assert state[1] == approx(1 / sqrt(2))


def test_substate_two_vars(qstate):
    state = qstate.substate([1, 0])

    assert state is not qstate
    assert len(state.variables) == 2

    assert state.variables[0].num_states == 2
    assert state.variables[1].num_states == 3

    assert state.variables[0].name == "b"
    assert state.variables[1].name == "a"

    assert list(state) == approx([0, 0, 0, 1 / sqrt(2), 0, 1 / sqrt(2)])


def test_substate_all_vars(qstate):
    state = qstate.substate([0, 1, 2])

    assert state is not qstate
    assert len(state.variables) == len(qstate.variables)
    assert list(state) == approx(list(qstate))


def test_substate_all_vars_permutations(qstate):
    state = qstate.substate([2, 0, 1])

    assert state is not qstate
    assert len(state.variables) == len(qstate.variables)

    one_half = approx(1 / 2)

    assert state[0, 0, 0] == 0
    assert state[1, 0, 0] == 0
    assert state[0, 0, 1] == one_half
    assert state[1, 0, 1] == one_half
    assert state[0, 1, 0] == 0
    assert state[1, 1, 0] == 0
    assert state[0, 1, 1] == 0
    assert state[1, 1, 1] == 0
    assert state[0, 2, 0] == 0
    assert state[1, 2, 0] == 0
    assert state[0, 2, 1] == one_half
    assert state[1, 2, 1] == one_half


def test_normalization():
    state = QuantumState.from_amplitudes([1, 1])
    assert list(state) == approx([1 / sqrt(2) + 0j, 1 / sqrt(2) + 0j])
