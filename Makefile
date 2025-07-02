.PHONY: test test-slicer test-gui install-deps clean find-slicer-python debug-python build-testing

# Default target
all: test

# Find Slicer's Python executable
find-slicer-python:
	@echo "Finding Slicer's Python executable..."
	@if command -v Slicer >/dev/null 2>&1; then \
		echo "Slicer found in PATH"; \
	elif [ -f "/Applications/Slicer.app/Contents/MacOS/Slicer" ]; then \
		echo "Slicer found at /Applications/Slicer.app/Contents/MacOS/Slicer"; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		echo "Slicer found at /usr/local/bin/Slicer"; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/PythonSlicer" ]; then \
		echo "Slicer found at $$SLICER_HOME/bin/PythonSlicer"; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/Slicer" ]; then \
		echo "Slicer found at $$SLICER_HOME/bin/Slicer"; \
	else \
		echo "ERROR: Slicer not found. Please install Slicer or add it to your PATH."; \
		exit 1; \
	fi

# Debug Python execution
debug-python:
	@echo "=== Python Execution Debug ==="
	@echo "System Python: $(shell which python)"
	@echo "System Python version: $(shell python --version 2>&1)"
	@echo ""
	@echo "Slicer Python:"
	@echo "  /Applications/Slicer.app/Contents/bin/PythonSlicer"
	@echo ""
	@echo "CTest will run tests in Slicer's Python environment."
	@echo "=================================="

# Build with testing enabled
build-testing: find-slicer-python
	@echo "Building with testing enabled..."
	@if [ ! -d "build" ]; then \
		mkdir build; \
	fi
	cd build && cmake -DBUILD_TESTING=ON ..
	cd build && make -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run Slicer-native tests (recommended for Slicer extensions)
test-slicer: build-testing
	@echo "Running Slicer-native tests..."
	cd build && ctest -V

# Run GUI tests (requires display and user interaction simulation)
test-gui: find-slicer-python
	@echo "Running GUI tests (requires display)..."
	@if [ -f "/Applications/Slicer.app/Contents/MacOS/Slicer" ]; then \
		/Applications/Slicer.app/Contents/MacOS/Slicer --python-script AnnotateUltrasound/Testing/Python/AnnotateUltrasoundGUITest.py; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script AnnotateUltrasound/Testing/Python/AnnotateUltrasoundGUITest.py; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/Slicer" ]; then \
		$$SLICER_HOME/bin/Slicer --python-script AnnotateUltrasound/Testing/Python/AnnotateUltrasoundGUITest.py; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/Slicer" ]; then \
		$$SLICER_HOME/Slicer --python-script AnnotateUltrasound/Testing/Python/AnnotateUltrasoundGUITest.py; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		echo "Checked paths:"; \
		echo "  /Applications/Slicer.app/Contents/MacOS/Slicer"; \
		echo "  /usr/local/bin/Slicer"; \
		echo "  $$SLICER_HOME/bin/Slicer"; \
		echo "  $$SLICER_HOME/Slicer"; \
		exit 1; \
	fi

# Run DICOM loading tests (requires display and real DICOM data)
test-dicom: find-slicer-python
	@echo "Running DICOM loading tests (requires display)..."
	@if [ -f "/Applications/Slicer.app/Contents/MacOS/Slicer" ]; then \
		/Applications/Slicer.app/Contents/MacOS/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/Slicer" ]; then \
		$$SLICER_HOME/bin/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/Slicer" ]; then \
		$$SLICER_HOME/Slicer --python-script AnnotateUltrasound/Testing/Python/test_dicom_loading.py; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		echo "Checked paths:"; \
		echo "  /Applications/Slicer.app/Contents/MacOS/Slicer"; \
		echo "  /usr/local/bin/Slicer"; \
		echo "  $$SLICER_HOME/bin/Slicer"; \
		echo "  $$SLICER_HOME/Slicer"; \
		exit 1; \
	fi

# Run all tests using CTest (Slicer-native)
test: test-pytest

# Build and install the module into Slicer
install-module: find-slicer-python
	@echo "Building and installing SlicerUltrasound module..."
	@python3 install_module.py

# Run pytest-style tests in Slicer Python environment
test-pytest: find-slicer-python
	@echo "Running pytest-style tests in Slicer Python environment..."
	@if [ -f "/Applications/Slicer.app/Contents/bin/PythonSlicer" ]; then \
		/Applications/Slicer.app/Contents/bin/PythonSlicer run_slicer_tests.py --install-deps; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script run_slicer_tests.py --install-deps; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/PythonSlicer" ]; then \
		$$SLICER_HOME/bin/PythonSlicer run_slicer_tests.py --install-deps; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		exit 1; \
	fi

# Run tests with specific pattern
test-pattern: build-testing
	@echo "Usage: make test-pattern PATTERN=AnnotateUltrasound"
	@if [ -z "$(PATTERN)" ]; then echo "Please specify PATTERN parameter"; exit 1; fi
	cd build && ctest -V -R $(PATTERN)

# Run pytest tests with coverage (Python-only)
test-coverage: find-slicer-python
	@echo "Running pytest tests with coverage..."
	@if [ -f "/Applications/Slicer.app/Contents/bin/PythonSlicer" ]; then \
		/Applications/Slicer.app/Contents/bin/PythonSlicer run_slicer_tests.py --install-deps --pytest-args --cov=. --cov-report=term-missing; \
	elif [ -f "/usr/local/bin/Slicer" ]; then \
		/usr/local/bin/Slicer --python-script run_slicer_tests.py --install-deps --pytest-args --cov=. --cov-report=term-missing; \
	elif [ -n "$$SLICER_HOME" ] && [ -f "$$SLICER_HOME/bin/PythonSlicer" ]; then \
		$$SLICER_HOME/bin/PythonSlicer run_slicer_tests.py --install-deps --pytest-args --cov=. --cov-report=term-missing; \
	else \
		echo "❌ Slicer not found. Please install Slicer first."; \
		exit 1; \
	fi

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf build/

# Help target
help:
	@echo "Available targets:"
	@echo "  find-slicer-python - Check if Slicer is available"
	@echo "  debug-python      - Show Python execution details"
	@echo "  build-testing     - Build with testing enabled"
	@echo "  install-module    - Build and install module into Slicer"
	@echo "  test-pytest       - Run pytest-style tests in Slicer Python environment"
	@echo "  test-gui          - Run GUI tests (requires display, simulates user interactions)"
	@echo "  test-dicom        - Run DICOM loading tests (requires display, uses real DICOM data)"
	@echo "  test              - Run pytest tests (alias for test-pytest)"
	@echo "  test-pattern      - Run tests matching pattern (e.g., make test-pattern PATTERN=AnnotateUltrasound)"
	@echo "  test-coverage     - Run pytest tests with coverage (Python-only)"
	@echo "  clean             - Clean up generated files"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "Note: This project uses Python-only modules, so we focus on Python testing."
	@echo "      test-pytest uses pytest in Slicer's Python environment"
	@echo "      test-gui uses Slicer's GUI test harness (requires display)"
	@echo "      test-dicom tests module with real DICOM data and annotations"
	@echo "      All tests run in Slicer's Python environment"