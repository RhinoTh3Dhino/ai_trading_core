.PHONY: clean test

# Rens cache og pyc-filer
clean:
	@echo "🧹 Rydder cache og pyc-filer..."
	@if exist .pytest_cache rmdir /S /Q .pytest_cache
	@for /R %%i in (*.pyc) do del /Q %%i
	@if exist __pycache__ rmdir /S /Q __pycache__

# Rens + geninstaller vigtige pakker + kør tests med coverage
test: clean
	@echo "📦 Installerer nødvendige pakker..."
	pip install --upgrade --force-reinstall "numpy<2.0" pandas_ta matplotlib kiwisolver
	@echo "🧪 Kører pytest med coverage..."
	pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing
