# math-llm

Building LLMs from scratch — math foundations to fine-tuned models.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## Usage

```bash
make test        # run tests
make coverage    # test + coverage report
make lint        # ruff linting
make typecheck   # mypy type checking
make clean       # remove build artifacts
```

## Structure

```
src/math_llm/    # library source code
tests/           # pytest test suite
```
