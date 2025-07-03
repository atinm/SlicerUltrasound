#!/usr/bin/env python3
"""
Test script to verify that the cache clearing is working correctly
"""

import slicer

def test_cache_clearing():
    """Test that the markup cache is properly cleared when reloading"""

    print("=== TESTING CACHE CLEARING ===")

    try:
        # Get the module widget and logic
        moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()
        moduleLogic = moduleWidget.logic

        print("✓ Module widget and logic found")

        # Check initial cache state
        print(f"\n1. Initial Cache State:")
        print(f"   - _lastMarkupFrameIndex: {getattr(moduleLogic, '_lastMarkupFrameIndex', 'Not set')}")
        print(f"   - _lastMarkupFrameHash: {getattr(moduleLogic, '_lastMarkupFrameHash', 'Not set')}")

        # Check if we have loaded data
        if not moduleLogic.sequenceBrowserNode:
            print("\n2. No sequence loaded - please load data first")
            return

        # Test cache clearing by calling clearScene
        print(f"\n3. Testing clearScene() cache clearing...")
        moduleLogic.clearScene()

        print(f"   - _lastMarkupFrameIndex after clearScene: {getattr(moduleLogic, '_lastMarkupFrameIndex', 'Not set')}")
        print(f"   - _lastMarkupFrameHash after clearScene: {getattr(moduleLogic, '_lastMarkupFrameHash', 'Not set')}")

        # Test cache clearing by calling onReadInputButton (if we have the setup)
        if moduleWidget.ui.inputDirectoryButton.directory and moduleWidget._parameterNode.rater:
            print(f"\n4. Testing onReadInputButton() cache clearing...")

            # Store current cache values
            old_index = getattr(moduleLogic, '_lastMarkupFrameIndex', None)
            old_hash = getattr(moduleLogic, '_lastMarkupFrameHash', None)

            # Call the reload function
            moduleWidget.onReadInputButton()

            # Check new cache values
            new_index = getattr(moduleLogic, '_lastMarkupFrameIndex', None)
            new_hash = getattr(moduleLogic, '_lastMarkupFrameHash', None)

            print(f"   - Cache cleared by onReadInputButton: {old_index != new_index or old_hash != new_hash}")
            print(f"   - New _lastMarkupFrameIndex: {new_index}")
            print(f"   - New _lastMarkupFrameHash: {new_hash}")
        else:
            print(f"\n4. Skipping onReadInputButton test - missing input directory or rater name")

        print("\n✓ Cache clearing test completed")

    except Exception as e:
        print(f"✗ Error during cache clearing test: {e}")

def force_cache_clear():
    """Manually clear the cache to test the fix"""
    print("\n=== MANUALLY CLEARING CACHE ===")

    try:
        moduleLogic = slicer.modules.annotateultrasound.widgetRepresentation().self().logic

        print("Current cache state:")
        print(f"  - _lastMarkupFrameIndex: {getattr(moduleLogic, '_lastMarkupFrameIndex', 'Not set')}")
        print(f"  - _lastMarkupFrameHash: {getattr(moduleLogic, '_lastMarkupFrameHash', 'Not set')}")

        # Manually clear the cache
        if hasattr(moduleLogic, '_lastMarkupFrameIndex'):
            moduleLogic._lastMarkupFrameIndex = None
        if hasattr(moduleLogic, '_lastMarkupFrameHash'):
            moduleLogic._lastMarkupFrameHash = None

        print("Cache cleared manually")
        print(f"  - _lastMarkupFrameIndex: {getattr(moduleLogic, '_lastMarkupFrameIndex', 'Not set')}")
        print(f"  - _lastMarkupFrameHash: {getattr(moduleLogic, '_lastMarkupFrameHash', 'Not set')}")

        # Force update to test if it works
        print("\nForcing updateLineMarkups()...")
        moduleLogic.updateLineMarkups()

        print("✓ Manual cache clearing completed")

    except Exception as e:
        print(f"✗ Error during manual cache clearing: {e}")

if __name__ == "__main__":
    # Run the test
    test_cache_clearing()

    # Ask if user wants to manually clear cache
    print("\nWould you like to manually clear the cache? (y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        force_cache_clear()