PYTHON	= uv run python
MAIN	= -m src
ARGS	?=

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
			 --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean fclean lint

install:
	uv sync

run:
	$(PYTHON) $(MAIN) $(ARGS)

debug:
	$(PYTHON) -m pdb $(MAIN) $(ARGS)

clean:
# delete every __pycache__ dir, except those of the packages installed in .venv
# ({} = the paths found, + = pass them all to a single rm call)
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

fclean: clean
	rm -rf .venv

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)
