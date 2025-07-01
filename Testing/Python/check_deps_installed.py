#!/usr/bin/env python3
"""
Script to check if dependencies are already installed in Slicer's Python environment.
If not, install them. This script should be run via: Slicer --python-script check_deps_installed.py
"""

import sys
import os
import slicer

def check_dependencies():
    """Check if all required dependencies are installed."""
    print(f"Checking dependencies in Slicer's Python: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Essential testing packages
    essential_packages = [
        "pytest",
        "pytest_cov",
        "pytest_mock"
    ]

    missing_packages = []
    for package in essential_packages:
        try:
            __import__(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
        except Exception as e:
            print(f"❌ Error checking {package}: {e}")
            missing_packages.append(package)

    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages."""
    if not missing_packages:
        print("✅ All dependencies are already installed!")
        return True

    print(f"\nInstalling missing packages: {missing_packages}")

    # Check if slicer.util.pip_install is available
    if not hasattr(slicer.util, 'pip_install'):
        print("❌ slicer.util.pip_install is not available")
        return False

    success_count = 0
    for package in missing_packages:
        try:
            print(f"Installing {package}...")
            slicer.util.pip_install(package)
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")

    print(f"\nInstallation summary: {success_count}/{len(missing_packages)} packages installed successfully")
    return success_count == len(missing_packages)

if __name__ == "__main__":
    print("=== Checking Dependencies in Slicer's Python Environment ===")

    missing_packages = check_dependencies()

    if missing_packages:
        success = install_missing_packages(missing_packages)
        if success:
            print("\n✅ All dependencies are now installed! Quitting Slicer...")
        else:
            print("\n❌ Some dependencies could not be installed.")
            print("   You may need to install them manually or use: make force-install-deps")
    else:
        print("\n✅ All dependencies are already installed!")

    # Quit Slicer
    slicer.util.quit()