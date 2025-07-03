#!/usr/bin/env python3
"""
Debug script to help identify why overlays and lines aren't appearing after onReadInputButton()
"""

import slicer
import logging

def test_overlays():
    """Debug function to check the state of overlays and lines after onReadInputButton()"""

    print("=== DEBUGGING OVERLAY/LINE VISIBILITY ISSUE ===")

    # Get the module widget and logic
    try:
        moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()
        moduleLogic = moduleWidget.logic
        parameterNode = moduleLogic.getParameterNode()
        print(f"✓ Module widget and logic found")
    except Exception as e:
        print(f"✗ Error getting module widget/logic: {e}")
        return

    # Check if DICOM data is loaded
    print(f"\n1. DICOM Data Status:")
    print(f"   - dfLoaded: {parameterNode.dfLoaded}")
    print(f"   - dicomDf exists: {moduleLogic.dicomDf is not None}")
    if moduleLogic.dicomDf is not None:
        print(f"   - Number of DICOM files: {len(moduleLogic.dicomDf)}")
        print(f"   - Current index: {moduleLogic.nextDicomDfIndex}")

    # Check sequence browser
    print(f"\n2. Sequence Browser Status:")
    print(f"   - sequenceBrowserNode exists: {moduleLogic.sequenceBrowserNode is not None}")
    if moduleLogic.sequenceBrowserNode:
        print(f"   - Selected item number: {moduleLogic.sequenceBrowserNode.GetSelectedItemNumber()}")
        print(f"   - Number of items: {moduleLogic.sequenceBrowserNode.GetNumberOfItems()}")

    # Check input and overlay volumes
    print(f"\n3. Volume Status:")
    print(f"   - inputVolume exists: {parameterNode.inputVolume is not None}")
    print(f"   - overlayVolume exists: {parameterNode.overlayVolume is not None}")

    if parameterNode.overlayVolume:
        displayNode = parameterNode.overlayVolume.GetDisplayNode()
        print(f"   - Overlay display node exists: {displayNode is not None}")
        if displayNode:
            print(f"   - Overlay window: {displayNode.GetWindow()}")
            print(f"   - Overlay level: {displayNode.GetLevel()}")

    # Check annotations
    print(f"\n4. Annotations Status:")
    print(f"   - annotations exists: {moduleLogic.annotations is not None}")
    if moduleLogic.annotations:
        print(f"   - frame_annotations count: {len(moduleLogic.annotations.get('frame_annotations', []))}")
        if 'frame_annotations' in moduleLogic.annotations:
            current_frame = moduleLogic.sequenceBrowserNode.GetSelectedItemNumber() if moduleLogic.sequenceBrowserNode else -1
            frame_data = next((f for f in moduleLogic.annotations['frame_annotations']
                             if f.get('frame_number') == current_frame), None)
            if frame_data:
                print(f"   - Current frame pleura lines: {len(frame_data.get('pleura_lines', []))}")
                print(f"   - Current frame b-lines: {len(frame_data.get('b_lines', []))}")
            else:
                print(f"   - No frame data for current frame {current_frame}")

    # Check markup nodes
    print(f"\n5. Markup Nodes Status:")
    print(f"   - pleuraLines count: {len(moduleLogic.pleuraLines)}")
    print(f"   - bLines count: {len(moduleLogic.bLines)}")

    visible_pleura = 0
    visible_blines = 0
    for i, node in enumerate(moduleLogic.pleuraLines):
        if node and node.GetDisplayNode():
            if node.GetDisplayNode().GetVisibility():
                visible_pleura += 1
                print(f"   - Pleura line {i}: visible, points: {node.GetNumberOfControlPoints()}")

    for i, node in enumerate(moduleLogic.bLines):
        if node and node.GetDisplayNode():
            if node.GetDisplayNode().GetVisibility():
                visible_blines += 1
                print(f"   - B-line {i}: visible, points: {node.GetNumberOfControlPoints()}")

    print(f"   - Visible pleura lines: {visible_pleura}")
    print(f"   - Visible b-lines: {visible_blines}")

    # Check rater selection
    print(f"\n6. Rater Selection Status:")
    print(f"   - Current rater: '{parameterNode.rater}'")
    print(f"   - selectedRaters: {getattr(moduleLogic, 'selectedRaters', 'Not set')}")
    print(f"   - seenRaters: {getattr(moduleLogic, 'seenRaters', 'Not set')}")

    # Check overlay visibility button
    print(f"\n7. UI Status:")
    try:
        overlay_button = moduleWidget.ui.overlayVisibilityButton
        print(f"   - Overlay visibility button checked: {overlay_button.checked}")
    except:
        print(f"   - Could not access overlay visibility button")

    # Check slice composite node
    print(f"\n8. Slice Composite Node Status:")
    try:
        redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
        print(f"   - Foreground volume ID: {redSliceCompositeNode.GetForegroundVolumeID()}")
        print(f"   - Foreground opacity: {redSliceCompositeNode.GetForegroundOpacity()}")
        print(f"   - Compositing: {redSliceCompositeNode.GetCompositing()}")
    except Exception as e:
        print(f"   - Error accessing slice composite node: {e}")

    # Check if overlay volume has data
    print(f"\n9. Overlay Volume Data:")
    if parameterNode.overlayVolume:
        try:
            overlayArray = slicer.util.arrayFromVolume(parameterNode.overlayVolume)
            print(f"   - Overlay array shape: {overlayArray.shape}")
            print(f"   - Overlay array min/max: {overlayArray.min()}/{overlayArray.max()}")
            print(f"   - Non-zero pixels: {overlayArray.sum()}")
        except Exception as e:
            print(f"   - Error accessing overlay array: {e}")

    print(f"\n=== END DEBUG INFO ===")

def force_update_overlay():
    """Force update the overlay to see if that helps"""
    print("\n=== FORCING OVERLAY UPDATE ===")

    try:
        moduleWidget = slicer.modules.annotateultrasound.widgetRepresentation().self()
        moduleLogic = moduleWidget.logic

        # Force update line markups
        print("Forcing updateLineMarkups()...")
        moduleLogic.updateLineMarkups()

        # Force update overlay volume
        print("Forcing updateOverlayVolume()...")
        moduleLogic.updateOverlayVolume()

        # Force overlay visibility
        print("Forcing overlay visibility...")
        moduleWidget.overlayVisibilityToggled(True)

        print("✓ Forced updates completed")

    except Exception as e:
        print(f"✗ Error during forced update: {e}")

if __name__ == "__main__":
    # Run the debug function
    debug_overlay_issue()

    # Ask if user wants to force update
    print("\nWould you like to force update the overlay? (y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        force_update_overlay()