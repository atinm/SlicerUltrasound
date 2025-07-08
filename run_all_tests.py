#!/usr/bin/env python3
"""
Run all unit tests for SlicerUltrasound project.

This script runs tests from all module directories in the correct order
to avoid import conflicts.

Usage:
    python run_all_tests.py [pytest-arguments]

Examples:
    python run_all_tests.py                    # Run tests without coverage
    python run_all_tests.py --cov             # Run tests with coverage
    python run_all_tests.py --cov --cov-report=html  # Run with coverage and HTML report
"""

import subprocess
import sys
import os
import time
import argparse

def run_tests_in_directory(directory, description, pytest_args):
    """Run tests in a specific directory."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"Directory: {directory}")
    print(f"{'='*60}")

    start_time = time.time()

    # Build pytest command with arguments
    pytest_cmd = [sys.executable, "-m", "pytest", "-v"] + pytest_args

    if directory == "tests":
        # Run from root directory
        pytest_cmd.append("tests/")
        result = subprocess.run(pytest_cmd, capture_output=False)
    else:
        # Change to module directory and run tests
        original_dir = os.getcwd()
        try:
            module_dir = directory.split('/')[0]
            os.chdir(module_dir)
            pytest_cmd.append("tests/")
            result = subprocess.run(pytest_cmd, capture_output=False)
        finally:
            os.chdir(original_dir)

    end_time = time.time()
    duration = end_time - start_time

    if result.returncode == 0:
        print(f"\n✅ {description} PASSED ({duration:.2f}s)")
        return True
    else:
        print(f"\n❌ {description} FAILED ({duration:.2f}s)")
        return False

def run_common_tests(pytest_args):
    """Run tests in AnonymizeUltrasound/common/tests/."""
    print(f"\n{'='*60}")
    print(f"Running AnonymizeUltrasound Common Tests")
    print(f"Directory: AnonymizeUltrasound/common/tests/")
    print(f"{'='*60}")

    start_time = time.time()
    pytest_cmd = [sys.executable, "-m", "pytest", "-v"] + pytest_args + ["AnonymizeUltrasound/common/tests/"]
    result = subprocess.run(pytest_cmd, capture_output=False)
    end_time = time.time()
    duration = end_time - start_time

    if result.returncode == 0:
        print(f"\n✅ AnonymizeUltrasound Common Tests PASSED ({duration:.2f}s)")
        return True
    else:
        print(f"\n❌ AnonymizeUltrasound Common Tests FAILED ({duration:.2f}s)")
        return False

def main():
    """Run all unit tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run all unit tests for SlicerUltrasound project")
    parser.add_argument("pytest_args", nargs="*", help="Arguments to pass to pytest")
    args = parser.parse_args()

    print("SlicerUltrasound Unit Test Runner")
    print("=" * 60)
    if args.pytest_args:
        print(f"Pytest arguments: {' '.join(args.pytest_args)}")

    overall_start = time.time()
    results = []

    # Test directories and descriptions
    test_suites = [
        ("AnnotateUltrasound/tests", "AnnotateUltrasound Module Tests"),
        ("AnonymizeUltrasound/tests", "AnonymizeUltrasound Module Tests"),
        ("tests", "Core/Shared Tests"),
    ]

    # Run each test suite
    for directory, description in test_suites:
        success = run_tests_in_directory(directory, description, args.pytest_args)
        results.append((description, success))

    # Run common tests separately
    success = run_common_tests(args.pytest_args)
    results.append(("AnonymizeUltrasound Common Tests", success))

    # Summary
    overall_end = time.time()
    total_duration = overall_end - overall_start

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed} test suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Total Tests: {passed + failed} unit tests")

    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n💥 {failed} TEST SUITE(S) FAILED 💥")
        return 1

if __name__ == "__main__":
    sys.exit(main())