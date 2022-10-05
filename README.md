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
app to read the documentation of the supported command line arguments. `-i`
allows you to specify the number of steps that the machine should perform; `-d`
will print the superposition of the machine after each step.

