#!/usr/bin/env python3
"""
Test runner for SlicerUltrasound module tests.
This script runs tests in Slicer's Python environment.
"""

import sys
import os
import argparse

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def install_dependencies():
    """Install test dependencies using Slicer's pip_install utility."""
    try:
        import slicer
        print("Checking test dependencies...")
        packages = ['pytest', 'pytest-cov', 'pytest-mock']

        for package in packages:
            try:
                # Check if package is already installed
                __import__(package.replace('-', '_'))
                print(f"✓ {package} already installed")
            except ImportError:
                # Try to install if not found
                print(f"Installing {package}...")
                slicer.util.pip_install(package)
                print(f"✓ Installed {package}")
        return True
    except Exception as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def run_tests(args):
    """Run tests using Slicer's Python environment."""
    try:
        import pytest

        # Use absolute path for tests directory
        tests_dir = os.path.join(SCRIPT_DIR, 'tests')

        # Change to the script directory to ensure proper test discovery
        original_cwd = os.getcwd()
        os.chdir(SCRIPT_DIR)

        test_args = [
            '--cov=.',
            '--cov-report=term-missing',
            '--tb=short',
            '--ignore=AnnotateUltrasound/Testing/Python/',
            '--ignore=Testing/Python/',
            tests_dir
        ]

        # Add any additional pytest arguments
        if args.pytest_args:
            test_args.extend(args.pytest_args)

        print(f"Running tests with: pytest {' '.join(test_args)}")
        print(f"Tests directory: {tests_dir}")
        print(f"Working directory: {os.getcwd()}")

        # Run pytest
        result = pytest.main(test_args)

        # Restore original working directory
        os.chdir(original_cwd)

        return result
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Run SlicerUltrasound tests in Slicer Python environment')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install test dependencies')
    parser.add_argument('--pytest-args', nargs='*', default=[],
                       help='Additional arguments to pass to pytest')

    args = parser.parse_args()

    print(f"Script directory: {SCRIPT_DIR}")

    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            return 1

    # Run tests
    return run_tests(args)

if __name__ == '__main__':
    sys.exit(main())