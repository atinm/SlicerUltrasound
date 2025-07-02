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
        print("✓ Slicer module imported successfully")
        print(f"Python executable: {sys.executable}")
        print(f"Slicer module location: {slicer.__file__}")
        print(f"Slicer module attributes: {dir(slicer)}")

        # Check if slicer.util exists
        if hasattr(slicer, 'util'):
            print("✓ slicer.util exists")
            print(f"slicer.util attributes: {dir(slicer.util)}")

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
        else:
            print("✗ slicer.util does not exist, using subprocess fallback")
            return install_dependencies_subprocess()

    except Exception as e:
        print(f"✗ Failed to install dependencies: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

def install_dependencies_subprocess():
    """Install test dependencies using subprocess calls to Slicer Python."""
    import subprocess

    # Find Slicer Python executable
    slicer_python = None
    possible_paths = [
        "/Applications/Slicer.app/Contents/bin/PythonSlicer",
        "/usr/local/bin/Slicer",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            slicer_python = path
            break

    if not slicer_python:
        print("✗ Could not find Slicer Python executable")
        return False

    print("Checking test dependencies...")
    packages = ['pytest', 'pytest-cov', 'pytest-mock']

    for package in packages:
        try:
            # Check if package is already installed
            result = subprocess.run([slicer_python, '-c', f'import {package.replace("-", "_")}'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {package} already installed")
            else:
                # Try to install if not found
                print(f"Installing {package}...")
                subprocess.run([slicer_python, '-m', 'pip', 'install', package],
                             check=True, capture_output=True)
                print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

    return True

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

def run_tests_subprocess(args):
    """Run tests using subprocess calls to Slicer Python."""
    import subprocess

    # Find Slicer Python executable
    slicer_python = None
    possible_paths = [
        "/Applications/Slicer.app/Contents/bin/PythonSlicer",
        "/usr/local/bin/Slicer",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            slicer_python = path
            break

    if not slicer_python:
        print("✗ Could not find Slicer Python executable")
        return 1

    # Use absolute path for tests directory
    tests_dir = os.path.join(SCRIPT_DIR, 'tests')

    test_args = [
        slicer_python, '-m', 'pytest',
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

    print(f"Running tests with: {' '.join(test_args)}")
    print(f"Tests directory: {tests_dir}")

    try:
        # Change to the script directory to ensure proper test discovery
        original_cwd = os.getcwd()
        os.chdir(SCRIPT_DIR)

        result = subprocess.run(test_args, check=False)

        # Restore original working directory
        os.chdir(original_cwd)

        return result.returncode
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
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Try to run in Slicer's environment first
    try:
        import slicer
        print("✓ Running in Slicer's Python environment")
        print(f"Slicer module location: {slicer.__file__}")

        # Install dependencies if requested
        if args.install_deps:
            if not install_dependencies():
                return 1

        # Run tests
        return run_tests(args)

    except ImportError as e:
        print(f"⚠️  Slicer module not available, using subprocess fallback...")
        print(f"ImportError: {e}")

        # Install dependencies if requested
        if args.install_deps:
            if not install_dependencies():
                return 1

        # Run tests using subprocess
        return run_tests_subprocess(args)

if __name__ == '__main__':
    sys.exit(main())