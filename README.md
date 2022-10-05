# Quantum Turing machine simulator in Python

## Reuqirements

* Python 3.10 or later.

* `poetry` for virtualenv management (optional).

## Installation

### Local installation with `poetry`

1. Check out this repository to your machine.

2. Run `poetry install` to install all required dependencies in an isolated
   environment.

3. Run `poetry run pyqtm` to run the application.

### Local installation witout `poetry`

1. Check out this repository to your machine.

2. Create a local virtual environment with `python -m venv .venv`

3. Run `.venv/bin/pip install -r requirements.txt` to install the dependencies
   that are needed to run the app.

4. Run `.venv/bin/pip install -e .` to install the application itself in the
   virtualenv.

5. Run `.venv/bin/pyqtm` to start the application.

### Using Docker

1. Run `docker build -t pyqtm:latest .` to build a Docker image.

2. Run `docker run -it --rm -v "$(pwd):/data" pyqtm:latest` to run the application,
   mounting the current folder to `/data`. This allows you to reach the
   examples from inside the container like this:
   `docker run -it --rm -v "$(pwd):/data" pyqtm:latest /data/doc/examples/deutsch_problem_constant.json`

## Running the examples

Several examples are provided in `doc/examples`; you can run these by passing
the name of the JSON file to the application. See also the `-h` switch of the
app to read the documentation of the supported command line arguments. `-n`
allows you to specify the number of steps that the machine should perform; `-d`
will print the superposition of the machine after each step; `-i` allows you to
override the default input of the machine from the specification file.

## Machine specifications

Quantum Turing machines are specified with JSON files; see the example files in
`doc/examples` for inspiration. Typically, a machine specification has two
mandatory and one optional top-level key:

* `description` (optional) provides a textual description of what the machine
  is supposed to be doing.

* `tape` encodes the default contents of the tape when the machine starts, as
  a string. The number of cells on the tape is inferred from the length of the
  string. Use `_` to represent blank cells and chars from `0` to `9` to
  represent symbols.

* `rules` encodes the transition rules of the machine as a JSON array of
  arrays. Each item in the `rules` array must contain the following items in
  the following order:

  - current state of the machine (integer, 0-based)
  - current symbol on the tape (integer, 0-based, use `"_"` for a blank symbol)
  - next state of the machine (integer, 0-based)
  - symbol to write on the tape (integer, 0-based, use `"_"` for a blank symbol)
  - head movement (`"left"`, `"right"` or `"stay"`)
  - probability amplitude of this rule as an integer or as a complex number.
    Complex numbers are represented with strings in the format "X+Yj".

To make it easier to create machine specifications, the following rules are
also applied:

* When the next state of the machine is omitted or `null`, it is the same as
  the current state.
* When the symbol to write is omitted or `null`, the current symbol on the tape
  is left intact.
* When the head movement command is omitted, it defaults to `"stay"`.
* When the probability amplitude is omitted, it defaults to 1. You can easily
  create deterministic Turing machines by omitting probability amplitudes and
  ensuring that each current state and symbol combination appears only once.
* Probability amplitudes are normalized by the app so you don't need to specify
  values like `1 + sqrt(2)` exactly -- just use a probability amplitude of 1
  for multiple rules and they will automatically be normalized appropriately.
  See `doc/examples/hadamard.json` for an example.
* In the `rules` array, you can add arbitrary strings as comments; these will
  be ignored by the parser.

## Caveats

Most naive specifications of quantum Turing machines do not result in an
unitary time evolution matrix (especially when you are trying to translate
a classical Turing machine). The app will warn about non-unitary transition
matrices, but it will still try to run the dynamics. If you get nonsensical
results, this means that the machine has somehow transitioned into a state
where the time evolution matrix did not preseve the length of the state vector.

