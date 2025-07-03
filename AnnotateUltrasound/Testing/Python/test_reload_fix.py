#!/usr/bin/env python3
"""
Test script to verify that the reload fix works correctly
"""

import slicer
import time

def test_reload_functionality():
    """Test the reload functionality to ensure overlays appear correctly"""

    print("=== TESTING RELOAD FUNCTIONALITY ===")

    try:
        # Get the module widget
        moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()
        moduleLogic = moduleWidget.logic
        parameterNode = moduleLogic.getParameterNode()

        print("✓ Module widget found")

        # Check initial state
        print(f"\n1. Initial State:")
        print(f"   - dfLoaded: {parameterNode.dfLoaded}")
        print(f"   - sequenceBrowserNode exists: {moduleLogic.sequenceBrowserNode is not None}")
        print(f"   - overlayVolume exists: {parameterNode.overlayVolume is not None}")

        # Simulate first load (if not already loaded)
        if not parameterNode.dfLoaded:
            print("\n2. Simulating first load...")
            # You would need to set up the input directory and rater name here
            print("   - Please set up input directory and rater name manually")
            print("   - Then click 'Read Input' button")
            return

        # Test reload
        print("\n3. Testing reload...")
        print("   - Calling onReadInputButton()...")

        # Store current state
        old_sequence_browser = moduleLogic.sequenceBrowserNode
        old_overlay_volume = parameterNode.overlayVolume

        # Call the reload function
        moduleWidget.onReadInputButton()

        # Wait a moment for processing
        time.sleep(1)

        # Check new state
        print(f"\n4. After Reload:")
        print(f"   - dfLoaded: {parameterNode.dfLoaded}")
        print(f"   - sequenceBrowserNode exists: {moduleLogic.sequenceBrowserNode is not None}")
        print(f"   - overlayVolume exists: {parameterNode.overlayVolume is not None}")
        print(f"   - sequenceBrowserNode changed: {old_sequence_browser != moduleLogic.sequenceBrowserNode}")
        print(f"   - overlayVolume changed: {old_overlay_volume != parameterNode.overlayVolume}")

        # Check if overlay is visible
        if parameterNode.overlayVolume:
            try:
                redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
                foreground_volume_id = redSliceCompositeNode.GetForegroundVolumeID()
                print(f"   - Foreground volume ID: {foreground_volume_id}")
                print(f"   - Overlay volume ID: {parameterNode.overlayVolume.GetID()}")
                print(f"   - Overlay is foreground: {foreground_volume_id == parameterNode.overlayVolume.GetID()}")
            except Exception as e:
                print(f"   - Error checking overlay visibility: {e}")

        # Check markup nodes
        print(f"\n5. Markup Nodes:")
        print(f"   - pleuraLines count: {len(moduleLogic.pleuraLines)}")
        print(f"   - bLines count: {len(moduleLogic.bLines)}")

        visible_pleura = sum(1 for node in moduleLogic.pleuraLines if node and node.GetDisplayNode() and node.GetDisplayNode().GetVisibility())
        visible_blines = sum(1 for node in moduleLogic.bLines if node and node.GetDisplayNode() and node.GetDisplayNode().GetVisibility())

        print(f"   - Visible pleura lines: {visible_pleura}")
        print(f"   - Visible b-lines: {visible_blines}")

        print("\n✓ Reload test completed")

    except Exception as e:
        print(f"✗ Error during reload test: {e}")

def force_reload_test():
    """Force a reload test by calling the button directly"""
    print("\n=== FORCING RELOAD TEST ===")

    try:
        moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()

        # Check if we have the necessary setup
        if not moduleWidget.ui.inputDirectoryButton.directory:
            print("✗ No input directory selected")
            return

        if not moduleWidget._parameterNode.rater:
            print("✗ No rater name entered")
            return

        print("✓ Input directory and rater name are set")
        print("   - Input directory:", moduleWidget.ui.inputDirectoryButton.directory)
        print("   - Rater name:", moduleWidget._parameterNode.rater)

        # Force the reload
        print("\nForcing reload...")
        moduleWidget.onReadInputButton()

        print("✓ Reload completed")

    except Exception as e:
        print(f"✗ Error during forced reload: {e}")

if __name__ == "__main__":
    # Run the test
    test_reload_functionality()

    # Ask if user wants to force reload
    print("\nWould you like to force a reload test? (y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        force_reload_test()