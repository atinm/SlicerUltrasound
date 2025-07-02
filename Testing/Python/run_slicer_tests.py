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

        # Debug: Print Slicer module search paths and loaded modules
        print("\n=== SLICER MODULE DEBUG INFO ===")
        try:
            print("Slicer version:", slicer.app.applicationVersion())
            print("Available modules:", slicer.moduleNames())
            print("AnnotateUltrasound in modules?", "AnnotateUltrasound" in slicer.moduleNames())
            print("Hasattr slicer.modules.AnnotateUltrasound?", hasattr(slicer.modules, "AnnotateUltrasound"))
        except Exception as e:
            print(f"Error getting module info: {e}")

        # Debug: Check Slicer log for errors
        print("\n=== SLICER LOG CHECK ===")
        try:
            import tempfile
            log_dir = tempfile.gettempdir()
            print(f"Checking for Slicer logs in: {log_dir}")
            import glob
            log_files = glob.glob(os.path.join(log_dir, "*.log")) + glob.glob(os.path.join(log_dir, "Slicer*.log"))
            if log_files:
                print(f"Found log files: {log_files}")
                for log_file in log_files[:3]:  # Check first 3 log files
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                            if 'error' in content.lower() or 'fail' in content.lower():
                                print(f"Log file {log_file} contains errors:")
                                lines = content.split('\n')
                                for line in lines[-20:]:  # Last 20 lines
                                    if 'error' in line.lower() or 'fail' in line.lower():
                                        print(f"  {line}")
                    except Exception as e:
                        print(f"Could not read log file {log_file}: {e}")
            else:
                print("No log files found")
        except Exception as e:
            print(f"Error checking logs: {e}")

        print("=== END DEBUG INFO ===\n")

        # Debug: Try to find and load AnnotateUltrasound module
        print("=== ANNOTATEULTRASOUND MODULE CHECK ===")
        try:
            import os
            slicer_root = os.path.dirname(os.path.dirname(slicer.__file__))
            slicer_version = slicer.app.applicationVersion().split(' ')[0]  # e.g., '5.8.0'
            major_minor = '.'.join(slicer_version.split('.')[:2])  # '5.8'
            common_paths = [
                os.path.join(slicer_root, f"lib/Slicer-{major_minor}/qt-scripted-modules"),
                os.path.join(slicer_root, f"share/Slicer-{major_minor}/qt-scripted-modules"),
            ]
            for path in common_paths:
                annotate_path = os.path.join(path, 'AnnotateUltrasound')
                print(f"Checking {annotate_path} ...")
                if os.path.exists(annotate_path):
                    print(f"  ✓ Found at: {annotate_path}")
                    print(f"    Contents: {os.listdir(annotate_path)}")
                else:
                    print(f"  ✗ Not found at: {annotate_path}")
        except Exception as e:
            print(f"Error checking AnnotateUltrasound module: {e}")
        print("=== END ANNOTATEULTRASOUND CHECK ===\n")

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

    # Check environment variable first (for CI environments)
    slicer_home = os.environ.get('SLICER_HOME')
    if slicer_home:
        possible_paths = [
            os.path.join(slicer_home, 'bin', 'PythonSlicer'),
            os.path.join(slicer_home, 'bin', 'Slicer'),
            os.path.join(slicer_home, 'PythonSlicer'),
            os.path.join(slicer_home, 'Slicer'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                slicer_python = path
                print(f"Found Slicer Python at: {slicer_python}")
                break

    # Fallback to common installation paths
    if not slicer_python:
        possible_paths = [
            "/Applications/Slicer.app/Contents/bin/PythonSlicer",
            "/usr/local/bin/Slicer",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                slicer_python = path
                print(f"Found Slicer Python at: {slicer_python}")
                break

    if not slicer_python:
        print("✗ Could not find Slicer Python executable")
        print("Checked paths:")
        if slicer_home:
            print(f"  {slicer_home}/bin/PythonSlicer")
            print(f"  {slicer_home}/bin/Slicer")
            print(f"  {slicer_home}/PythonSlicer")
            print(f"  {slicer_home}/Slicer")
        print("  /Applications/Slicer.app/Contents/bin/PythonSlicer")
        print("  /usr/local/bin/Slicer")
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

        # Use absolute path for tests directory (project root)
        project_root = os.path.dirname(os.path.dirname(SCRIPT_DIR))
        tests_dir = os.path.join(project_root, 'tests')

        # Change to the project root to ensure proper test discovery
        original_cwd = os.getcwd()
        os.chdir(project_root)

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

    # Check environment variable first (for CI environments)
    slicer_home = os.environ.get('SLICER_HOME')
    if slicer_home:
        possible_paths = [
            os.path.join(slicer_home, 'bin', 'PythonSlicer'),
            os.path.join(slicer_home, 'bin', 'Slicer'),
            os.path.join(slicer_home, 'PythonSlicer'),
            os.path.join(slicer_home, 'Slicer'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                slicer_python = path
                print(f"Found Slicer Python at: {slicer_python}")
                break

    # Fallback to common installation paths
    if not slicer_python:
        possible_paths = [
            "/Applications/Slicer.app/Contents/bin/PythonSlicer",
            "/usr/local/bin/Slicer",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                slicer_python = path
                print(f"Found Slicer Python at: {slicer_python}")
                break

    if not slicer_python:
        print("✗ Could not find Slicer Python executable")
        print("Checked paths:")
        if slicer_home:
            print(f"  {slicer_home}/bin/PythonSlicer")
            print(f"  {slicer_home}/bin/Slicer")
            print(f"  {slicer_home}/PythonSlicer")
            print(f"  {slicer_home}/Slicer")
        print("  /Applications/Slicer.app/Contents/bin/PythonSlicer")
        print("  /usr/local/bin/Slicer")
        return 1

    # Use absolute path for tests directory (project root)
    project_root = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    tests_dir = os.path.join(project_root, 'tests')

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
        # Change to the project root to ensure proper test discovery
        original_cwd = os.getcwd()
        os.chdir(project_root)

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