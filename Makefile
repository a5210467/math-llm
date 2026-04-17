.PHONY: install test lint typecheck clean coverage

install:
	pip install -e ".[dev]" 2>/dev/null || pip install -e . && pip install -r requirements.txt

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=math_llm --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; \
	rm -rf .coverage htmlcov/ .mypy_cache/ .pytest_cache/
