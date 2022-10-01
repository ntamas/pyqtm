"""Command-line interface for the quantum Turing machine simulator, main
entry point.
"""

from argparse import ArgumentParser
from json import load
from typing import List, Optional, Tuple

import sys

from pyqtm.model import QuantumTuringMachine

from .formatting import format_amplitude, format_machine_state


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="pyqtm")
    parser.add_argument(
        "filename",
        help="name of the Turing machine specification file",
        metavar="FILENAME",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="initial state of the tape; may be used to override the initial state of the tape specified in the machine specification file",
        default=None,
        required=False,
        metavar="INPUT",
    )
    parser.add_argument(
        "-n",
        "--num-steps",
        help="number of steps to simulate",
        metavar="STEPS",
        default=50,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--detailed",
        help="show the execution of the machine step by step",
        default=False,
        action="store_true",
    )
    return parser


def _main(
    filename: str,
    input: Optional[str] = None,
    num_steps: int = 50,
    detailed: bool = False,
) -> int:
    with open(filename) as fp:
        spec = load(fp)

    if input:
        spec["tape"] = input

    machine = QuantumTuringMachine.from_json(spec)
    schedule = ([1] * (num_steps + 1)) if detailed else [num_steps]
    total_steps = 0

    for steps_to_take in schedule:
        if detailed:
            print(f"T={total_steps}")

        for state, amplitude in machine.state.iter_states(sort=True, reverse=True):
            if abs(amplitude) >= 1e-4:
                print(format_machine_state(state), "  P:", format_amplitude(amplitude))

        total_steps += steps_to_take
        machine.simulate(steps_to_take)

        if detailed:
            print("-" * 50)

    return 0


def main() -> int:
    options = create_parser().parse_args()
    sys.exit(_main(**vars(options)))
