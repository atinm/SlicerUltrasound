import csv
import json
import logging
import math
import numpy as np
import os
import glob
import pydicom
import qt
import shutil
import slicer
import vtk
import colorsys
import copy
import re
import zlib
import datetime

# Import AnnotateUltrasound base module
import AnnotateUltrasound as annotate

try:
    import pandas as pd
except ImportError:
    slicer.util.pip_install('pandas')
    import pandas as pd

try:
    import cv2
except ImportError:
    slicer.util.pip_install('opencv-python')
    import cv2

from collections import defaultdict
from DICOMLib import DICOMUtils
from typing import Annotated, Optional

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVectorVolumeNode
from slicer import vtkMRMLNode

#
# AdjudicateUltrasound
#

class AdjudicateUltrasound(annotate.AnnotateUltrasound):
    """AdjudicateUltrasound module, subclassing AnnotateUltrasound.AnnotateUltrasound."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Adjudicate Ultrasound"
        self.parent.categories = ["Ultrasound"]
        self.parent.dependencies = []
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]
        self.parent.helpText = f"""
This module facilitates the process of creating segmentations of B-lines and the pleura in series of B-mode lung ultrasound videos.<br><br>

See more information in <a href="https://github.com/SlicerUltrasound/SlicerUltrasound/blob/main/README.md">README</a> <a href="https://github.com/SlicerUltrasound/SlicerUltrasound/tree/main/AdjudicateUltrasound">Source Code</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Tamas Ungi (Queen's University), with support from MLSC Bits to Bytes grant for Point of Care Ultrasound, and NIH grants R21EB034075 and R01EB035679.
"""
        slicer.app.connect("startupCompleted()", postModuleDiscoveryTasks)

#
# Register sample data sets in Sample Data module
#

def postModuleDiscoveryTasks():
    """
    Performs initialization tasks after Slicer has been fully loaded.
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.
    pass


#
# AdjudicateUltrasoundParameterNode
#

@parameterNodeWrapper
class AdjudicateUltrasoundParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    overlayVolume: vtkMRMLVectorVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    invertedVolume: vtkMRMLScalarVolumeNode
    lineBeingPlaced: vtkMRMLNode = None
    dfLoaded: bool = False
    pleuraPercentage: float = -1.0
    unsavedChanges: bool = False
    depthGuideVisible: bool = True
    rater = ''
    showInvalidAndDuplicate: bool = True

#
# AdjudicateUltrasoundWidget
#
#
# global singleton instance of the widget
adjudicateUltrasoundWidgetInstance = None
def getAdjudicateUltrasoundWidget():
    """
    Get the singleton instance of the AdjudicateUltrasoundWidget.
    """
    global adjudicateUltrasoundWidgetInstance
    if adjudicateUltrasoundWidgetInstance is None:
        raise RuntimeError("AdjudicateUltrasoundWidget instance is not initialized")
    return adjudicateUltrasoundWidgetInstance

class AdjudicateUltrasoundWidget(annotate.AnnotateUltrasoundWidget):
    """AdjudicateUltrasoundWidget, subclassing AnnotateUltrasound.AnnotateUltrasoundWidget."""
    def __init__(self, parent=None) -> None:
        # Set up logic to point to AdjudicateUltrasoundLogic
        self.logic = AdjudicateUltrasoundLogic()

        super().__init__(parent)

        # Flag to track if the user manually expanded the rater table
        self._userManuallySetRaterTableState = False
        self._lastUserManualCollapsedState = None  # Track the last state the user manually set

    def resourcePath(self, *args):
        import os
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AnnotateUltrasound', 'Resources'))
        return os.path.join(base_path, *args)

    def initializeShortcuts(self):
        super().initializeShortcuts()

    def _createAdjudicationShortcuts(self):
        # Remove old shortcuts if needed
        for name in dir(self):
            if name.startswith("adjudicationShortcut"):
                try:
                    getattr(self, name).activated.disconnect()
                    getattr(self, name).setParent(None)
                    delattr(self, name)
                except Exception:
                    pass

        def make_shortcut(attr_name, key, slot):
            shortcut = qt.QShortcut(qt.QKeySequence(key), slicer.util.mainWindow())
            shortcut.setContext(qt.Qt.ApplicationShortcut)
            shortcut.activated.connect(slot)
            setattr(self, attr_name, shortcut)

        make_shortcut("adjudicationShortcutV", "V", self.onValidateLine)
        make_shortcut("adjudicationShortcutI", "I", self.onInvalidateLine)
        make_shortcut("adjudicationShortcutU", "U", self.onDuplicateLine)
        make_shortcut("adjudicationShortcutN", "N", self.selectNextUnadjudicatedLine)
        make_shortcut("adjudicationShortcutP", "P", self.selectPreviousUnadjudicatedLine)
        make_shortcut("adjudicationShortcutShiftN", "Shift+N", self.selectNextVisibleLine)
        make_shortcut("adjudicationShortcutShiftP", "Shift+P", self.selectPreviousVisibleLine)
        make_shortcut("adjudicationShortcutR", "R", self.onResetAllAdjudication)

    def connectKeyboardShortcuts(self):
        super().connectKeyboardShortcuts()
        self._createAdjudicationShortcuts()

    def disconnectKeyboardShortcuts(self):
        super().disconnectKeyboardShortcuts()
        # Disconnect AdjudicateUltrasound-specific shortcuts
        if hasattr(self, "_adjudicationShortcuts"):
            for shortcut in self._adjudicationShortcuts:
                shortcut.activated.disconnect()
                shortcut.setParent(None)
            self._adjudicationShortcuts.clear()

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        super().setup()

        # Update directory button directory from settings
        self.ui.inputDirectoryButton.directory = slicer.app.settings().value("AdjudicateUltrasound/InputDirectory", "")

        # Reconnect readInputButton, inputDirectoryButton, saveButton, and saveAndLoadNextButton
        # to the overridden methods
        for name, method in {
            "readInputButton": self.onReadInputButton,
            "inputDirectoryButton": self.onInputDirectorySelected,
            "saveButton": self.onSaveButton,
            "saveAndLoadNextButton": self.onSaveAndLoadNextButton
        }.items():
            btn = getattr(self.ui, name, None)
            if not btn:
                continue
            try:
                if name == "inputDirectoryButton":
                    btn.directoryChanged.disconnect()
                    btn.directoryChanged.connect(method)
                else:
                    btn.clicked.disconnect()
                    btn.clicked.connect(method)
            except TypeError:
                # Not connected yet
                if name == "inputDirectoryButton":
                    btn.directoryChanged.connect(method)
                else:
                    btn.clicked.connect(method)

        # Create a new group box for adjudication tools and place it after raterColorsCollapsibleButton
        from slicer.util import findChildren
        parentLayout = self.ui.raterColorsCollapsibleButton.parent().layout()
        if not hasattr(self, "_adjudicationToolsGroupBox") or self._adjudicationToolsGroupBox is None:
            self._adjudicationShortcuts = []
            self._adjudicationToolsGroupBox = qt.QGroupBox("Adjudication Tools")
            self._adjudicationToolsGroupBox.setObjectName("adjudicationToolsGroupBox")
            mainLayout = qt.QVBoxLayout()
            mainLayout.setSpacing(4)
            mainLayout.setContentsMargins(4, 4, 4, 4)

            def make_button(label, tooltip, shortcutKey, slot):
                btn = qt.QPushButton(label)
                btn.setToolTip(tooltip)
                # Remove shortcut creation from here, handled globally now
                btn.clicked.connect(slot)
                return btn

            # Row 1
            row1 = qt.QHBoxLayout()
            row1.addWidget(make_button("Validate Line [V]", "Mark the selected line as validated (Shortcut V key)", "V", self.onValidateLine))
            row1.addWidget(make_button("Invalidate Line [I]", "Mark the selected line as invalidated (Shortcut I key)", "I", self.onInvalidateLine))
            row1.addWidget(make_button("Duplicate Line [U]", "Mark the selected line as duplicate (Shortcut U key)", "U", self.onDuplicateLine))
            mainLayout.addLayout(row1)

            # Row 2
            row2 = qt.QHBoxLayout()
            row2.addWidget(make_button("Next Line [N]", "Navigate to next line (Shortcut N key)", "N", self.selectNextUnadjudicatedLine))
            row2.addWidget(make_button("Prev Line [P]", "Navigate to previous line (Shortcut P key)", "P", self.selectPreviousUnadjudicatedLine))
            mainLayout.addLayout(row2)

            # Row 3
            row3 = qt.QHBoxLayout()
            row3.addWidget(make_button("Next Visible [Shift+N]", "Navigate to next visible annotation line (Shortcut Shift+N)", "Shift+N", self.selectNextVisibleLine))
            row3.addWidget(make_button("Prev Visible [Shift+P]", "Navigate to previous visible annotation line (Shortcut Shift+P)", "Shift+P", self.selectPreviousVisibleLine))
            mainLayout.addLayout(row3)

            # Row 4
            row4 = qt.QHBoxLayout()
            row4.addWidget(make_button("Validate Rest", "Mark all visible unadjudicated lines as validated", "", self.onValidateAllUnadjudicated))
            row4.addWidget(make_button("Invalidate Rest", "Mark all visible unadjudicated lines as invalidated", "", self.onInvalidateAllUnadjudicated))
            row4.addWidget(make_button("Duplicate Rest", "Mark all visible unadjudicated lines as duplicated", "", self.onDuplicateAllUnadjudicated))
            mainLayout.addLayout(row4)

            # Reset row
            resetRow = qt.QHBoxLayout()
            resetRow.addWidget(make_button("Reset All [R]", "Reset all lines to unadjudicated (Shortcut R key)", "R", self.onResetAllAdjudication))
            mainLayout.addLayout(resetRow)

            # Checkbox
            self.showInvalidatedCheckBox = qt.QCheckBox("Show Invalidated or Duplicated Lines")
            self.showInvalidatedCheckBox.setToolTip("Show Invalidated or Duplicated Lines")
            self.showInvalidatedCheckBox.stateChanged.connect(self.onShowInvalidOrDuplicateToggled)
            self.showInvalidatedCheckBox.setChecked(True)
            mainLayout.addWidget(self.showInvalidatedCheckBox)

            self._adjudicationToolsGroupBox.setLayout(mainLayout)
            index = parentLayout.indexOf(self.ui.raterColorsCollapsibleButton)
            parentLayout.insertWidget(index + 1, self._adjudicationToolsGroupBox)

        # After UI and logic setup, create adjudication shortcuts
        self.connectKeyboardShortcuts()

        # Add observer to selection node to enforce only visible nodes can be selected
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if not hasattr(self, 'selectionObserverTag') or self.selectionObserverTag is None:
            self.selectionObserverTag = selectionNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onSelectionChanged)

        # Connect rater table collapsed signal to detect user manual changes
        if hasattr(self.ui, 'raterColorsCollapsibleButton'):
            self.ui.raterColorsCollapsibleButton.connect('collapsedChanged(bool)', self.onRaterColorTableCollapsedChanged)

        # Ensure parameter node is initialized after setup is complete
        self.initializeParameterNode()
        # Load fixed labels file
        self.onLabelsFileSelected()


    # These functions are used to handle clicks on the red view when selecting lines by clicking on the red view
    # near the line to be selected.
    # --- Click observer on red view ---
    def setupClickObserverOnRedView(self):
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        renderWindowInteractor = sliceWidget.sliceView().renderWindow().GetInteractor()
        self._clickObserverTag = renderWindowInteractor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.onRedViewClick)

    # --- Distance point to segment ---
    def distancePointToSegment(self, p, a, b):
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        ab = b - a
        if np.allclose(ab, 0):
            return np.linalg.norm(p - a)
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    # --- On red view click ---
    # This function is called when the user clicks on the red view.
    # It finds the closest visible markup line node (to any segment) and selects it.
    # If the closest node is within 3 mm of the click, it selects the node.
    # If the closest node is more than 3 mm away, it does nothing.
    def onRedViewClick(self, caller, event):
        x, y = caller.GetEventPosition()
        # Get the slice widget and node
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        sliceNode = sliceWidget.mrmlSliceNode()
        xyToRas = sliceNode.GetXYToRAS()
        xy = [x, y, 0, 1]
        ras = [0, 0, 0, 1]
        xyToRas.MultiplyPoint(xy, ras)
        ras = ras[:3]
        # Find the closest visible markup line node (to any segment)
        minDist = float('inf')
        closestNode = None
        threshold = 3.0  # mm
        for node in slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode'):
            displayNode = node.GetDisplayNode()
            nodeRater = node.GetAttribute('rater')
            if not displayNode or not displayNode.GetVisibility():
                continue
            nPoints = node.GetNumberOfControlPoints()
            for i in range(nPoints - 1):
                pt1 = [0, 0, 0]
                pt2 = [0, 0, 0]
                node.GetNthControlPointPosition(i, pt1)
                node.GetNthControlPointPosition(i + 1, pt2)
                dist = self.distancePointToSegment(ras, pt1, pt2)
                if dist < minDist:
                    minDist = dist
                    closestNode = node
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if closestNode and minDist < threshold:
            selectionNode.SetActivePlaceNodeID(closestNode.GetID())
        else:
            # Clear selection if clicking away from any line
            selectionNode.SetActivePlaceNodeID("")
        # Update line markups to refresh visual appearance (highlighting, etc.)
        self.logic.updateLineMarkups()

    def onSelectionChanged(self, caller, event):
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        if selectedNodeID:
            node = slicer.mrmlScene.GetNodeByID(selectedNodeID)
            if node and node.GetClassName() == "vtkMRMLMarkupsLineNode":
                displayNode = node.GetDisplayNode()
                if not displayNode or not displayNode.GetVisibility():
                    selectionNode.SetActivePlaceNodeID("")  # Clear selection
        # Update line markups to refresh visual appearance (highlighting, etc.)
        self.logic.updateLineMarkups()

    def saveUserSettings(self):
        settings = qt.QSettings()
        settings.setValue('AdjudicateUltrasound/ShowPleuraPercentage', self.ui.showPleuraPercentageCheckBox.checked)
        settings.setValue('AdjudicateUltrasound/DepthGuide', self.ui.depthGuideCheckBox.checked)
        settings.setValue('AdjudicateUltrasound/Rater', self.ui.raterName.text.strip())
        ratio = self.logic.updateOverlayVolume()
        if ratio is not None:
            self._parameterNode.pleuraPercentage = ratio * 100
        self._updateGUIFromParameterNode()

    def _updateMarkupsAndOverlayProgrammatically(self, setUnsavedChanges=False):
        super()._updateMarkupsAndOverlayProgrammatically(setUnsavedChanges)

        # Clear selection if selected node is not visible
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        if selectedNodeID:
            node = slicer.mrmlScene.GetNodeByID(selectedNodeID)
            if node and node.GetClassName() == "vtkMRMLMarkupsLineNode":
                displayNode = node.GetDisplayNode()
                if not displayNode or not displayNode.GetVisibility():
                    selectionNode.SetActivePlaceNodeID("")  # Clear selection

    def onInputDirectorySelected(self):
        logging.info('onInputDirectorySelected')

        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = '⚠️ Please select an input directory'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            return

        # Update local settings
        slicer.app.settings().setValue("AdjudicateUltrasound/InputDirectory", inputDirectory)

    def onValidateLine(self):
        self._setLineValidationStatus("validated", "validate", "validated")

    def onInvalidateLine(self):
        self._setLineValidationStatus("invalidated", "invalidate", "invalidated")

    def onDuplicateLine(self):
        self._setLineValidationStatus("duplicated", "mark as duplicated", "marked as duplicated")

    def _setLineValidationStatus(self, status, action_verb, status_description):
        """
        Set the validation status of the selected line.
        Args:
            status: The validation status to set ("validated", "invalidated", or "duplicated")
            action_verb: The verb for the error message (e.g., "validate", "invalidate")
            status_description: The description for the success message (e.g., "validated", "invalidated")
        """
        # Get the selected markup node
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        if not selectedNodeID:
            slicer.util.showStatusMessage(f"No line selected to {action_verb}.", 3000)
            return
        markupNode = slicer.mrmlScene.GetNodeByID(selectedNodeID)
        if not markupNode or not markupNode.GetClassName() == "vtkMRMLMarkupsLineNode":
            slicer.util.showStatusMessage("Selected node is not a line.", 3000)
            return
        # Only allow validation if node is visible
        displayNode = markupNode.GetDisplayNode()
        if not displayNode or not displayNode.GetVisibility():
            slicer.util.showStatusMessage("Cannot adjudicate: selected line is not visible.", 3000)
            return
        # Get current rater (adjudicator)
        adjudicator = self.getCurrentRater()
        validation = {
            "status": status,
            "adjudicator": adjudicator,
            "timestamp": datetime.datetime.now().isoformat()
        }
        markupNode.SetAttribute("validation", json.dumps(validation))
        # Set opacity based on status
        if status == "validated":
            displayNode.SetOpacity(1.0)
        else:
            displayNode.SetOpacity(0.3)
        # Also update the corresponding annotation line
        self._updateAnnotationLineValidation(markupNode, validation)
        # Reset cache to force immediate color updates when validation status changes
        self.logic._resetMarkupCache()
        self.logic.updateCurrentFrame()
        slicer.util.showStatusMessage(f"Line {status_description}.", 3000)

    def onValidateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("validated", "Validated", 1.0)

    def onInvalidateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("invalidated", "Invalidated", 0.3)

    def onDuplicateAllUnadjudicated(self):
        self._setAllUnadjudicatedLinesStatus("duplicated", "Duplicated", 0.3)

    def _setAllUnadjudicatedLinesStatus(self, status, status_past_tense, opacity):
        """
        Set the validation status of all unadjudicated lines.
        Args:
            status: The validation status to set ("validated", "invalidated", or "duplicated")
            status_past_tense: The past tense form for the success message (e.g., "Validated", "Invalidated")
            opacity: The opacity to set for the lines (1.0 for validated, 0.3 for others)
        """
        # Get current rater (adjudicator)
        adjudicator = self.getCurrentRater()
        count = 0

        # Set programmatic update flag to prevent unsavedChanges from being set
        self.logic._isProgrammaticUpdate = True

        try:
            for nodeIndex in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsLineNode")):
                markupNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, "vtkMRMLMarkupsLineNode")
                displayNode = markupNode.GetDisplayNode()
                if not displayNode or not displayNode.GetVisibility():
                    continue # Skip if not visible
                validation_json = markupNode.GetAttribute("validation")
                current_status = None
                if validation_json:
                    try:
                        validation = json.loads(validation_json)
                        current_status = validation.get("status", None)
                    except Exception:
                        current_status = None
                if not current_status or current_status == "unadjudicated":
                    validation = {
                        "status": status,
                        "adjudicator": adjudicator,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    markupNode.SetAttribute("validation", json.dumps(validation))

                    # Update the annotation data for this line
                    self._updateAnnotationLineValidation(markupNode, validation)

                    count += 1

            # Reset cache to force immediate color updates when validation status changes
            self.logic._resetMarkupCache()

            # Update the visual appearance of all lines
            self.logic.updateLineMarkups()

            # Update the current frame to save the changes to annotations
            self.logic.updateCurrentFrame()

        finally:
            # Reset programmatic update flag
            self.logic._isProgrammaticUpdate = False

        # Set unsavedChanges since this is a user action that modifies validation status
        self._parameterNode.unsavedChanges = True

        slicer.util.showStatusMessage(f"{status_past_tense} {count} unadjudicated lines.", 3000)

    def onShowInvalidOrDuplicateToggled(self, checked):
        if self._parameterNode:
            self._parameterNode.showInvalidAndDuplicate = (checked != 0)
        for nodeIndex in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsLineNode")):
            markupNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, "vtkMRMLMarkupsLineNode")
            validation_json = markupNode.GetAttribute("validation")
            status = None
            if validation_json:
                try:
                    validation = json.loads(validation_json)
                    status = validation.get("status", None)
                except Exception:
                    status = None
            if status in ("invalidated", "duplicated"):
                displayNode = markupNode.GetDisplayNode()
                if displayNode:
                    if checked:
                        displayNode.SetVisibility(True)
                        displayNode.SetOpacity(0.3)
                    else:
                        displayNode.SetVisibility(False)
        slicer.util.showStatusMessage(f"{'Showing' if checked else 'Hiding'} invalidated/duplicated lines.", 3000)

    def getCurrentRater(self):
        # Return the current rater name from parameter node or UI
        if hasattr(self, '_parameterNode') and hasattr(self._parameterNode, 'rater'):
            return self._parameterNode.rater
        if hasattr(self.ui, 'raterName'):
            return self.ui.raterName.text().strip()
        return ""

    def _updateAnnotationLineValidation(self, markupNode, validation):
        """
        Update the validation field in self.logic.annotations for the line matching this markupNode in the current frame.
        """
        if self.logic.annotations is None or 'frame_annotations' not in self.logic.annotations:
            return
        if self.logic.sequenceBrowserNode is None:
            return
        currentFrameIndex = max(0, self.logic.sequenceBrowserNode.GetSelectedItemNumber())
        frame = next((item for item in self.logic.annotations['frame_annotations'] if int(item.get("frame_number", -1)) == currentFrameIndex), None)
        if not frame:
            return
        # Get rater and points from markupNode
        rater = markupNode.GetAttribute("rater")
        if not rater:
            logging.warning("Attempted to adjudicate a markup node with no rater set. Node name: %s, points: %s", markupNode.GetName(), [markupNode.GetNthControlPointPosition(i, [0,0,0]) for i in range(markupNode.GetNumberOfControlPoints())])
            return
        points = []
        for i in range(markupNode.GetNumberOfControlPoints()):
            coord = [0, 0, 0]
            markupNode.GetNthControlPointPosition(i, coord)
            points.append(coord)
        # Try to match in pleura_lines and b_lines
        for key in ["pleura_lines", "b_lines"]:
            for line in frame.get(key, []):
                if line.get("rater") == rater:
                    line_points = line.get("line", {}).get("points", [])
                    if len(line_points) == len(points) and all([all(abs(a-b)<1e-6 for a,b in zip(pt1,pt2)) for pt1,pt2 in zip(line_points, points)]):
                        line["validation"] = validation
                        return

    def _selectLineByFilter(self, filter_fn, direction, empty_message, status_message_prefix):
        # Gather all lines matching the filter in the current frame
        nodes = []
        for node in self.logic.pleuraLines + self.logic.bLines:
            if node.GetDisplayNode() and node.GetDisplayNode().GetVisibility() and filter_fn(node):
                nodes.append(node)
        if not nodes:
            slicer.util.showStatusMessage(empty_message, 3000)
            return
        # Find currently selected node
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID()
        try:
            idx = [n.GetID() for n in nodes].index(selectedNodeID)
            if direction == "next":
                new_idx = (idx + 1) % len(nodes)
            else:
                new_idx = (idx - 1) % len(nodes)
        except ValueError:
            new_idx = 0 if direction == "next" else len(nodes) - 1
        newNode = nodes[new_idx]
        selectionNode.SetActivePlaceNodeID(newNode.GetID())
        slicer.util.showStatusMessage(f"{status_message_prefix} {new_idx+1} of {len(nodes)}.", 2000)
        self.logic.updateLineMarkups()
        rater = newNode.GetAttribute("rater") or "unknown"
        line_type = "pleura" if newNode in self.logic.pleuraLines else "b-line"
        logging.info(f"{direction[0].upper()}: Selected {line_type} line from rater '{rater}' (line {new_idx+1} of {len(nodes)})")

    def _isUnadjudicated(self, node):
        """Check if a node is unadjudicated (no validation status or status is 'unadjudicated')."""
        validation_json = node.GetAttribute("validation")
        status = None
        if validation_json:
            try:
                validation = json.loads(validation_json)
                status = validation.get("status", None)
            except Exception:
                status = None
        return not status or status == "unadjudicated"

    def selectNextUnadjudicatedLine(self):
        self._selectLineByFilter(self._isUnadjudicated, "next", "No unadjudicated lines in this frame.", "Selected unadjudicated line")

    def selectPreviousUnadjudicatedLine(self):
        self._selectLineByFilter(self._isUnadjudicated, "previous", "No unadjudicated lines in this frame.", "Selected unadjudicated line")

    def selectNextVisibleLine(self):
        self._selectLineByFilter(lambda node: True, "next", "No visible lines in this frame.", "Selected visible line")

    def selectPreviousVisibleLine(self):
        self._selectLineByFilter(lambda node: True, "previous", "No visible lines in this frame.", "Selected visible line")

    def onShowHideLines(self, checked=None):
        """Toggle visibility of all lines and overlays."""
        if checked is None:
            # Toggle button state
            self.ui.showHideLinesButton.setChecked(not self.ui.showHideLinesButton.isChecked())
            checked = self.ui.showHideLinesButton.isChecked()
        # Set visibility of all lines
        for node in self.logic.pleuraLines + self.logic.bLines:
            displayNode = node.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(checked)
        # Also toggle overlay visibility
        self.ui.overlayVisibilityButton.setChecked(checked)

    def onResetAllAdjudication(self):
        """Reset all lines in the current frame to unadjudicated status."""
        for node in self.logic.pleuraLines + self.logic.bLines:
            displayNode = node.GetDisplayNode()
            if displayNode and displayNode.GetVisibility():
                node.SetAttribute("validation", json.dumps({"status": "unadjudicated"}))
                displayNode.SetOpacity(1.0)
        self.logic.updateCurrentFrame()
        self._parameterNode.unsavedChanges = True
        slicer.util.showStatusMessage("All lines reset to unadjudicated.", 3000)

    def extractSeenAndSelectedRaters(self):
        """
        Extracts the set of raters that have contributed lines in the current annotations,
        ensuring the current rater is included even if not present in any frame annotations.
        Sets self.seenRaters to a sorted list of rater names.
        """
        # Use the Logic's centralized method to extract and set up raters
        super().extractSeenAndSelectedRaters()

    def refocusAndRestoreShortcuts(self, delay: int = 200):
        qt.QTimer.singleShot(delay, self._delayedSetRedViewFocus)
        qt.QTimer.singleShot(delay + 100, self._restoreFocusAndShortcuts)

    def onReadInputButton(self):
        """
        Read the input directory and update the dicomDf dataframe, using rater-specific annotation files.

        :return: True if the input directory was read successfully, False otherwise.
        """
        logging.info('adjudicate: onReadInputButton')

        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            statusText = '⚠️ Please select an input directory'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            self._parameterNode.dfLoaded = False
            return

        rater = self._parameterNode.rater
        if not rater:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Missing Rater Name",
                "Please enter a rater name before loading the input directory."
            )
            self.ui.statusLabel.setText("⚠️ Please enter a rater name before loading.")
            self._parameterNode.dfLoaded = False
            return

        # Remove existing sequence browser observer before reloading
        if self.logic.sequenceBrowserNode:
            self.removeObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.onSequenceBrowserModified)

        numFilesFound = self.logic.updateInputDf(rater, inputDirectory)
        logging.info(f"Found {numFilesFound} DICOM files")
        statusText = f"Found {numFilesFound} DICOM files"

        if numFilesFound > 0:
            self._parameterNode.dfLoaded = True
            # Update progress bar
            self.ui.progressBar.minimum = 0
            self.ui.progressBar.maximum = numFilesFound
            self.ui.progressBar.value = 0
            waitDialog = self.createWaitDialog("Loading first sequence", "Loading first sequence...")

            # Set navigation flag to prevent rater table state changes
            self._isNavigating = True

            # Reset the DICOM index to start from the beginning when reloading
            self.logic.nextDicomDfIndex = 0

            self.currentDicomDfIndex = self.logic.loadNextSequence()

            # Add observer for the new sequence browser node
            if self.logic.sequenceBrowserNode:
                self.addObserver(self.logic.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.onSequenceBrowserModified)

            # Update self.ui.currentFileLabel using the DICOM file name
            currentDicomFilepath = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
            currentDicomFilename = os.path.basename(currentDicomFilepath)
            statusText = f"Current file ({self.logic.nextDicomDfIndex}/{len(self.logic.dicomDf)}): {currentDicomFilename}"
            self.ui.currentFileLabel.setText(statusText)
            self.ui.statusLabel.setText('')
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.logic.sequenceBrowserNode.SetSelectedItemNumber(0)
            self.logic.updateCurrentFrame()

            self.ui.intensitySlider.setValue(0)

            # After loading the first sequence, extract seen raters and update checkboxes
            self.extractSeenAndSelectedRaters()

            self._updateRaterColorTableCheckboxes()
            self.updateGuiFromAnnotations()

            # Close the wait dialog
            waitDialog.close()

            self.ui.progressBar.value = self.currentDicomDfIndex

            self.ui.overlayVisibilityButton.setChecked(True)

            # Mark that this is no longer the first DICOM load
            self._isFirstDicomLoad = False
        else:
            statusText = 'Could not find any files to load in input directory tree!'
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.ui.statusLabel.setText(statusText)

        self._updateGUIFromParameterNode()

        # Restore focus and shortcuts after loading input
        self.refocusAndRestoreShortcuts()

        # Clear navigation flag after all operations are complete
        self._isNavigating = False

    def confirmUnsavedChanges(self):
        """
        Asks the user if they want to save unsaved changes before continuing.
        """
        return super().confirmUnsavedChanges()


    def updateGuiFromAnnotations(self):
        super().updateGuiFromAnnotations()

    def saveAnnotations(self):
        """
        Saves current annotations to rater-specific json file.
        Returns True if save was successful, False otherwise.
        """
        try:
            # Add annotation line control points to the annotations dictionary and save it to file
            if self.logic.annotations is None:
                logging.error("saveAnnotations (adjudicate): No annotations loaded")
                return False

            # Check if rater name is set and not empty; if not, prompt user to enter one
            rater = self._parameterNode.rater
            if not rater:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Missing Rater Name",
                    "Rater name is not set. Please enter your rater name before saving."
                )
                self.ui.statusLabel.setText("⚠️ Please enter a rater name before saving.")
                return False

            waitDialog = self.createWaitDialog("Saving annotations", "Saving annotations...")

            # Check if any labels are checked
            annotationLabels = []
            for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())):
                widget = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
                if not isinstance(widget, qt.QGroupBox):
                    continue
                # Find all checkboxes in groupBox
                for j in reversed(range(widget.layout().count())):
                    checkBox = widget.layout().itemAt(j).widget()
                    if isinstance(checkBox, qt.QCheckBox) and checkBox.isChecked():
                        # Use original category/label for saving
                        origCategory = checkBox.property('originalCategory')
                        origLabel = checkBox.property('originalLabel')
                        annotationLabels.append(f"{origCategory}/{origLabel}")
            self.logic.annotations['labels'] = annotationLabels

            # We save all lines for adjudication
            inputDirectory = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['InputDirectory']
            file_path = self.logic.dicomDf.iloc[self.logic.nextDicomDfIndex - 1]['Filepath']
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            adjudication_file = os.path.join(inputDirectory, f"{base_name}.adjudication.json")
            # Save all lines (combined, no filtering) to adjudication file
            save_data = self.logic.convert_ras_to_lps(self.logic.annotations.get("frame_annotations", []))
            with open(adjudication_file, 'w') as f:
                json.dump(save_data, f)

            waitDialog.close()

            self._parameterNode.unsavedChanges = False

            statusText = f"✅ Adjudications saved successfully to {os.path.basename(adjudication_file)}"
            slicer.util.mainWindow().statusBar().showMessage(statusText, 3000)
            self.ui.statusLabel.setText(statusText)
            logging.info(f"Adjudications saved to {adjudication_file}")

            return True

        except Exception as e:
            statusText = f"❌ Failed to save adjudications: {str(e)}"
            slicer.util.mainWindow().statusBar().showMessage(statusText, 5000)
            self.ui.statusLabel.setText(statusText)
            logging.error(f"Error saving adjudications: {e}")
            return False

    def onSaveButton(self):
        """
        Saves current annotations to json file only
        """
        logging.info('onSaveButton (save)')
        success = self.saveAnnotations()
        if not success:
            # Error message already shown by saveAnnotations
            return

    def onSaveAndLoadNextButton(self):
        """
        Saves current annotations to json file and loads next sequence.
        """
        logging.info('onSaveAndLoadNextButton (save and load next scan)')

        success = self.saveAnnotations()
        if success:
            self.onNextButton()
            # After loading next, restore focus and shortcuts with a slight delay
            self.refocusAndRestoreShortcuts()
        else:
            # Error message already shown by saveAnnotations, don't proceed to next
            return

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        self.disconnectKeyboardShortcuts()  # Disconnect shortcuts when leaving the module

        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        widget = slicer.modules.adjudicateultrasound.widgetRepresentation()
        if widget and widget.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Guard: ensure logic is initialized
        if self.logic is None:
            logging.error("Logic not initialized before initializeParameterNode")
            return

        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
        if self.logic and self._parameterNode:
            self.logic.parameterNode = self._parameterNode

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        settings = slicer.app.settings()
        showDepthGuide = settings.value('AdjudicateUltrasound/DepthGuide', False)
        # be consistent and just read bool, convert if was string
        if isinstance(showDepthGuide, str):
            showDepthGuide = showDepthGuide.lower() == 'true'
        self._parameterNode.rater = settings.value('AdjudicateUltrasound/Rater', '')
        self.ui.raterName.setText(self._parameterNode.rater)
        if self._parameterNode.rater != '':
            self.logic.setRater(self._parameterNode.rater)
            self.logic.getColorsForRater(self._parameterNode.rater)
        self.ui.depthGuideCheckBox.setChecked(showDepthGuide)


    def setParameterNode(self, inputParameterNode: AdjudicateUltrasoundParameterNode) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self.logic and self._parameterNode:
            self.logic.parameterNode = self._parameterNode

        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGUIFromParameterNode)
            self._updateGUIFromParameterNode()

    def _updateGUIFromParameterNode(self, caller=None, event=None) -> None:
        if self.updatingGUI:
            return
        self.updatingGUI = True
        try:
            if self._parameterNode is None:
                logging.debug("No parameter node")
                return

            # Update line buttons
            if self._parameterNode.lineBeingPlaced is None:
                self.ui.addPleuraButton.setChecked(False)
                self.ui.addBlineButton.setChecked(False)
            else:
                if self._parameterNode.lineBeingPlaced.GetName() == "Pleura":
                    self.ui.addPleuraButton.setChecked(True)
                    self.ui.addBlineButton.setChecked(False)
                elif self._parameterNode.lineBeingPlaced.GetName() == "B-line":
                    self.ui.addPleuraButton.setChecked(False)
                    self.ui.addBlineButton.setChecked(True)
                else:
                    logging.error(f"Unknown line type {self._parameterNode.lineBeingPlaced.GetName()}")
                    return

            # If the frame index changed, then we want to make sure no row is selected in the frames table
            if self.logic.sequenceBrowserNode is not None:
                currentFrameIndex = self.logic.sequenceBrowserNode.GetSelectedItemNumber()
                if currentFrameIndex != self._lastFrameIndex:
                    self._lastFrameIndex = currentFrameIndex
                    self.ui.framesTableWidget.clearSelection()

            # Update corner annotation if _parameterNode.pleuraPercentage is a non-negative number
            if self.ui.showPleuraPercentageCheckBox.checked and self._parameterNode.pleuraPercentage >= 0:
                view=slicer.app.layoutManager().sliceWidget("Red").sliceView()
                view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft,f"B-line/Pleura = {self._parameterNode.pleuraPercentage:.1f} %")
                view.cornerAnnotation().GetTextProperty().SetColor(1,1,0)
                view.forceRender()

            # Update collapse/expand buttons
            if not self._parameterNode.dfLoaded:
                self.ui.inputsCollapsibleButton.collapsed = False
                self.ui.workflowCollapsibleButton.collapsed = True
                self.ui.sectorAnnotationsCollapsibleButton.collapsed = True
                self.ui.labelAnnotationsCollapsibleButton.collapsed = True
                # Collapse rater color table when no DICOM is loaded (no raters to display)
                if hasattr(self.ui, 'raterColorsCollapsibleButton'):
                    self._setRaterColorTableCollapsedState(True)
            else:
                self.ui.inputsCollapsibleButton.collapsed = True
                self.ui.workflowCollapsibleButton.collapsed = False
                self.ui.sectorAnnotationsCollapsibleButton.collapsed = False
                self.ui.labelAnnotationsCollapsibleButton.collapsed = False

                # Handle rater table collapse/expand logic
                if hasattr(self.ui, 'raterColorsCollapsibleButton'):
                    # During navigation, allow content updates but prevent collapse/expand state changes
                    if self._isNavigating:
                        # Skip collapse/expand logic during navigation, but still populate content
                        pass
                    else:
                        # If user manually set state, always restore it
                        if self._userManuallySetRaterTableState:
                            if self._lastUserManualCollapsedState is not None:
                                self._setRaterColorTableCollapsedState(self._lastUserManualCollapsedState)
                        else:
                            self._setRaterColorTableCollapsedState(True)

            # Save rater name to settings
            settings = qt.QSettings()
            settings.setValue('AdjudicateUltrasound/Rater', self.ui.raterName.text.strip())

            # Only update raterColorTable if present and DICOM is loaded
            if hasattr(self.ui, 'raterColorTable') and self._parameterNode.dfLoaded:
                self.populateRaterColorTable()
        finally:
            self.updatingGUI = False

    def _setRaterColorTableCollapsedState(self, collapsed):
        """
        Set the collapsed state of the rater color table.

        Args:
            collapsed: True to collapse, False to expand
        """
        if not hasattr(self.ui, 'raterColorsCollapsibleButton'):
            return

        self.ui.raterColorsCollapsibleButton.collapsed = collapsed

    def onRaterColorTableCollapsedChanged(self, collapsed):
        """
        Called when the user manually expands/collapses the rater table.
        Sets the flag to respect user's manual state.
        """
        self._userManuallySetRaterTableState = True
        self._lastUserManualCollapsedState = collapsed

    def populateRaterColorTable(self):
        if not hasattr(self.ui, 'raterColorTable'):
            return
        self.ui.raterColorTable.blockSignals(True)
        self.ui.raterColorTable.clearContents()
        colors = list(self.logic.getAllRaterColors())

        # Filter out __selected_node__, __adjudicated_node__ before setting row count, we don't want to show it in the UI
        visible_colors = [(r, (pleura_color, bline_color)) for r, (pleura_color, bline_color) in colors if r != "__selected_node__" and r != "__adjudicated_node__"]

        self.ui.raterColorTable.setRowCount(len(visible_colors))
        self.ui.raterColorTable.setColumnCount(3)
        self.ui.raterColorTable.setHorizontalHeaderLabels(["Rater", "Pleura", "B-line"])
        header = self.ui.raterColorTable.horizontalHeader()
        header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        # Columns 1 & 2: Color indicators — just enough to show the color
        header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)

        self.ui.raterColorTable.setColumnWidth(1, 30)
        self.ui.raterColorTable.setColumnWidth(2, 30)
        for row, (r, (pleura_color, bline_color)) in enumerate(visible_colors):
            rater_item = qt.QTableWidgetItem(r)
            rater_item.setFlags(qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
            if not hasattr(self, "selectedRaters") or r in self.selectedRaters:
                rater_item.setCheckState(qt.Qt.Checked)
            else:
                rater_item.setCheckState(qt.Qt.Unchecked)

            pleura_item = qt.QTableWidgetItem()
            pleura_item.setFlags(qt.Qt.ItemIsEnabled)
            pleura_item.setBackground(qt.QColor(*(int(c * 255) for c in pleura_color)))

            bline_item = qt.QTableWidgetItem()
            bline_item.setFlags(qt.Qt.ItemIsEnabled)
            bline_item.setBackground(qt.QColor(*(int(c * 255) for c in bline_color)))

            self.ui.raterColorTable.setItem(row, 0, rater_item)
            self.ui.raterColorTable.setItem(row, 1, pleura_item)
            self.ui.raterColorTable.setItem(row, 2, bline_item)
        self.ui.raterColorTable.blockSignals(False)

    def getSelectedRatersFromTable(self):
        selected = []
        table = self.ui.raterColorTable
        for row in range(table.rowCount):
            item = table.item(row, 0)
            if item is not None and item.checkState() == qt.Qt.Checked:
                selected.append(item.text())
        return selected

    def _updateRaterColorTableCheckboxes(self):
        """
        Helper function to update all checkboxes in the rater color table based on the selectedRaters.

        """

        if not hasattr(self.ui, 'raterColorTable'):
            return

        table = self.ui.raterColorTable
        table.blockSignals(True)
        try:
            for row in range(table.rowCount):
                item = table.item(row, 0)
                if item:
                    if item.text().strip().lower() in self.selectedRaters:
                        item.setCheckState(qt.Qt.Checked)
                    else:
                        item.setCheckState(qt.Qt.Unchecked)
        finally:
            table.blockSignals(False)
        self.ui.raterColorTable.repaint()
        self.ui.raterColorTable.update()

    def onRaterColorSelectionChangedFromUser(self):
        if self.updatingGUI:
            return
        self.updateRatersFromCheckboxes()

    def updateRatersFromCheckboxes(self):
        self.selectedRaters = self.getSelectedRatersFromTable()
        self.logic.setSelectedRaters(self.selectedRaters)
        self._updateMarkupsAndOverlayProgrammatically()
        self._updateGUIFromParameterNode()
        self.ui.raterColorTable.repaint()
        self.ui.raterColorTable.update()

    def onRaterColorTableClicked(self, row, column):
        item = self.ui.raterColorTable.item(row, 0)  # Assume checkbox is in column 0
        if item is not None:
            current_state = item.checkState()
            item.setCheckState(qt.Qt.Unchecked if current_state == qt.Qt.Checked else qt.Qt.Checked)
        self.onRaterColorSelectionChangedFromUser()

    def _getActiveSequenceBrowserNode(self):
        """Return the active sequence browser node, even if the toolbar is focused on a sequence node."""
        node = slicer.modules.sequences.toolBar().activeBrowserNode()
        if node is None:
            return None
        # If it's already a browser node, return it
        if isinstance(node, slicer.vtkMRMLSequenceBrowserNode):
            return node
        # Otherwise, find the browser node that references this sequence node
        sequenceBrowsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        for browser in sequenceBrowsers:
            collection = vtk.vtkCollection()
            browser.GetSynchronizedSequenceNodes(collection, True)
            if collection.IsItemPresent(node):
                return browser
        return None

    def _navigateToFrameInSequence(self, target_frame_calculator, already_at_message):
        """
        Generic frame navigation method that eliminates code duplication.

        Args:
            target_frame_calculator: Function that takes (current_index, max_index) and returns target_index
            already_at_message: Status message to show when already at the target position
        """
        activeBrowserNode = self._getActiveSequenceBrowserNode()
        if activeBrowserNode:
            currentIndex = activeBrowserNode.GetSelectedItemNumber()
            maxIndex = activeBrowserNode.GetNumberOfItems() - 1

            targetIndex = target_frame_calculator(currentIndex, maxIndex)

            if targetIndex != currentIndex:
                activeBrowserNode.SetSelectedItemNumber(targetIndex)
                # Reset selected node ID when changing frames
                selectionNode = slicer.app.applicationLogic().GetSelectionNode()
                if selectionNode:
                    selectionNode.SetActivePlaceNodeID("")
                self._setRedViewFocus()
            else:
                slicer.util.mainWindow().statusBar().showMessage(already_at_message, 3000)

    def _nextFrameInSequence(self):
        """Go to next frame in the current sequence using Slicer's built-in sequence browser."""
        def next_target(current, max_index):
            return current + 1 if current < max_index else current

        self._navigateToFrameInSequence(next_target, '⚠️ Already at last frame')

    def _previousFrameInSequence(self):
        """Go to previous frame in the current sequence using Slicer's built-in sequence browser."""
        def previous_target(current, max_index):
            return current - 1 if current > 0 else current

        self._navigateToFrameInSequence(previous_target, '⚠️ Already at first frame')

    def _firstFrameInSequence(self):
        """Go to the first frame in the current sequence."""
        def first_target(current, max_index):
            return 0 if current > 0 else current

        self._navigateToFrameInSequence(first_target, '⚠️ Already at first frame')

    def _lastFrameInSequence(self):
        """Go to the last frame in the current sequence."""
        def last_target(current, max_index):
            return max_index if current < max_index else current

        self._navigateToFrameInSequence(last_target, '⚠️ Already at last frame')

    def _togglePlayPauseSequence(self):
        """Toggle play/pause for the current sequence browser."""
        activeBrowserNode = self._getActiveSequenceBrowserNode()
        if activeBrowserNode:
            isPlaying = activeBrowserNode.GetPlaybackActive()
            activeBrowserNode.SetPlaybackActive(not isPlaying)
        self._setRedViewFocus()

    def _setRedViewFocus(self):
        """Set focus to the red view to ensure keyboard shortcuts work immediately."""
        # Use a timer to delay focus setting to ensure all UI updates are complete
        qt.QTimer.singleShot(200, self._delayedSetRedViewFocus)

    def _delayedSetRedViewFocus(self):
        """Delayed focus setting to ensure all UI updates are complete."""
        try:
            # Since shortcuts are connected to main window, focus on that
            mainWindow = slicer.util.mainWindow()
            if mainWindow:
                mainWindow.activateWindow()
                mainWindow.setFocus()
                mainWindow.raise_()

            # Also try setting focus to the module widget itself
            if hasattr(self, 'parent') and self.parent:
                self.parent.setFocus()

        except Exception as e:
            logging.warning(f"Could not set focus: {e}")

    def _forceShortcutsActive(self):
        """Force keyboard shortcuts to be active by temporarily disconnecting and reconnecting them."""
        try:
            # Temporarily disconnect and reconnect shortcuts to force them to be active
            self.disconnectKeyboardShortcuts()
            qt.QTimer.singleShot(50, self.connectKeyboardShortcuts)
        except Exception as e:
            logging.warning(f"Could not force shortcuts active: {e}")

    def _restoreFocusAndShortcuts(self):
        """Restore focus to main window and ensure shortcuts are active."""
        try:
            # Reset interaction mode to ensure keyboard shortcuts work
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            if interactionNode:
                interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

            # Set focus back to main window
            mainWindow = slicer.util.mainWindow()
            if mainWindow:
                mainWindow.setFocus()
                # After setting main window focus, also set focus to Red slice widget if available
                layoutManager = slicer.app.layoutManager()
                if layoutManager:
                    redWidget = layoutManager.sliceWidget("Red")
                    if redWidget:
                        redWidget.setFocus()

            # Force shortcuts to be active
            self._forceShortcutsActive()
        except Exception as e:
            logging.warning(f"Could not restore focus and shortcuts: {e}")

    def _navigateToClip(self, direction):
        """Helper method to navigate to previous or next clip."""
        direction_text = "previous" if direction == "previous" else "next"
        slicer.util.mainWindow().statusBar().showMessage(f'Loading {direction_text} clip...', 2000)
        if direction == "previous":
            self.onPreviousButton()
        else:
            self.onNextButton()

    def _onPageUpPressed(self):
        """Handle Page Up press for next clip."""
        self._navigateToClip("next")

    def _onPageDownPressed(self):
        """Handle Page Down press for previous clip."""
        self._navigateToClip("previous")

    def _onPreviousClipPressed(self):
        """Handle Shift+Up or Ctrl+Up press for previous clip."""
        self._navigateToClip("previous")

    def _onNextClipPressed(self):
        """Handle Shift+Down or Ctrl+Down press for next clip."""
        self._navigateToClip("next")

    def onSequenceBrowserModified(self, caller, event):
        """Handle sequence browser modifications (e.g., frame navigation via Slicer UI)."""
        # Reset selected node ID when sequence browser changes (UI state management)
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        if selectionNode:
            selectionNode.SetActivePlaceNodeID("")

        # Call Logic class to update markups (data processing)
        if self.logic:
            self.logic._updateMarkupsAndOverlayProgrammatically(None)

    def onShowHideLines(self, checked=None):
        """Toggle visibility of all lines and overlays."""
        if checked is None:
            # Toggle button state
            self.ui.showHideLinesButton.setChecked(not self.ui.showHideLinesButton.isChecked())
            checked = self.ui.showHideLinesButton.isChecked()
        # Set visibility of all lines
        for node in self.logic.pleuraLines + self.logic.bLines:
            displayNode = node.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(checked)
        # Also toggle overlay visibility
        self.ui.overlayVisibilityButton.setChecked(checked)

#
# AdjudicateUltrasoundLogic
#

class AdjudicateUltrasoundLogic(annotate.AnnotateUltrasoundLogic):
    """
    AdjudicateUltrasoundLogic, subclassing annotate.AnnotateUltrasoundLogic.
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        # These variables keep their values when the scene is cleared
        self.dicomDf = None
        self.nextDicomDfIndex = 0

        # These variables need to be reset when the scene is cleared
        self.annotations = None
        self.pleuraLines = []
        self.bLines = []
        self.sequenceBrowserNode = None
        self.depthGuideMode = 1
        self.seen_basenames = set()
        self.dcm_by_base = {}  # base_name → dcm filepath

        # Flag to track when we're doing programmatic updates (to avoid setting unsavedChanges)
        self._isProgrammaticUpdate = False

    # Static variable to track seen raters and their order
    seenRaters = []
    realRaters = []
    selectedRaters = []

    def _getOrCreateParameterNode(self):
        if not hasattr(self, "parameterNode"):
            baseWrapper = super()._getOrCreateParameterNode()
            rawNode = baseWrapper.parameterNode  # unwrap
            self.parameterNode = AdjudicateUltrasoundParameterNode(rawNode)
        return self.parameterNode

    def getParameterNode(self):
        return self._getOrCreateParameterNode()

    def getCurrentRater(self):
        return self.getParameterNode().rater.strip().lower()

    def getColorsForRater(self, rater: str):
        """
        Assign unique, visually distinct colors for pleura and b-lines per rater.
        Each rater gets completely unique colors that are distinct both within the rater and between raters.
        Uses golden ratio distribution starting from green (pleura) and blue (b-lines).
        """

        rater = rater.strip().lower()
        current_rater = self.getCurrentRater()

        if rater not in self.seenRaters and rater != '':
            self.seenRaters.append(rater)
            self.seenRaters.sort()

        if current_rater not in self.seenRaters:
            self.seenRaters.append(current_rater)
            self.seenRaters.sort()

        raters = self.seenRaters

        if rater not in raters:
            return [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]  # fallback red/orange

        N = len(raters)

        if N == 0:
            return [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]  # fallback red/orange

        # Find the index of this rater among all non-current raters
        rater_index = raters.index(rater)

        # Use golden ratio for non-repeating distribution
        # φ = (1 + √5) / 2 ≈ 1.618033988749895
        golden_ratio = (1 + 5**0.5) / 2

        # Start from positions after green and blue to avoid conflicts with current rater
        pleura_start = 0.66 # Start at blue
        bline_start = 0.33  # Start at green

        # Generate hues by adding golden ratio steps from the starting points
        pleura_hue = (pleura_start + rater_index * golden_ratio) % 1.0
        bline_hue = (bline_start + rater_index * golden_ratio) % 1.0

        # Ensure pleura and b-line colors are distinct by adjusting saturation and value
        pleura_rgb = colorsys.hsv_to_rgb(pleura_hue, 0.95, 1.0)  # bright colors
        bline_rgb = colorsys.hsv_to_rgb(bline_hue, 0.95, 1.0)   # bright colors
        return list(pleura_rgb), list(bline_rgb)

    def getAllRaterColors(self):
        """
        Returns a list of (rater, (pleura_color, bline_color)) for all seen raters.
        """
        colors = []
        for r in self.seenRaters:
            pleura_color, bline_color = self.getColorsForRater(r)
            colors.append((r, (pleura_color, bline_color)))
        return colors

    def setSelectedRaters(self, raters: set):
        """
        Store the selected raters and filter visuals accordingly.
        """
        self.selectedRaters = set(raters)

    def getSelectedRaters(self):
        if hasattr(self, "selectedRaters"):
            return self.selectedRaters
        return None

    def setRater(self, value):
        node = self.getParameterNode()
        wasModifying = node.StartModify()
        node.rater = value.strip().lower()
        node.EndModify(wasModifying)

    def updateInputDf(self, rater, input_folder):
        """
        Update the dicomDf dataframe with the DICOM files in the input folder.

        :param input_folder: The input folder to search for DICOM files.
        :return: The number of rows in the dataframe and the number of annotations files created.
        """
        logging.info('adjudicate: updateInputDf')

        dicom_data = []

        # Recursively walk through the input folder
        for dcm_path in glob.glob(f"{input_folder}/**/*.dcm", recursive=True):
            base = os.path.splitext(os.path.basename(dcm_path))[0]
            if base not in self.seen_basenames:
                self.seen_basenames.add(base)
                self.dcm_by_base[base] = dcm_path
            else:
                continue

        # Get the total number of files
        total_files = len(self.dcm_by_base)

        # Create a QProgressDialog
        progress_dialog = qt.QProgressDialog("Parsing DICOM files...", "Cancel", 0, total_files)
        progress_dialog.setWindowModality(qt.Qt.WindowModal)
        progress_dialog.show()

        sorted_dcm_by_base = dict(sorted(self.dcm_by_base.items()))
        self.dcm_by_base = sorted_dcm_by_base

        file_count = 0
        for key, file_path in self.dcm_by_base.items():
            progress_dialog.setValue(file_count)
            slicer.app.processEvents()

            file_count += 1
            try:
                # Try to read the file as a DICOM file
                dicom_file = pydicom.dcmread(file_path)

                # Skip non-ultrasound modalities
                if dicom_file.get("Modality", "") != "US":
                    continue

                # Extract required information
                patient_uid = dicom_file.PatientID if 'PatientID' in dicom_file else None
                study_uid = dicom_file.StudyInstanceUID if 'StudyInstanceUID' in dicom_file else None
                series_uid = dicom_file.SeriesInstanceUID if 'SeriesInstanceUID' in dicom_file else None
                instance_uid = dicom_file.SOPInstanceUID if 'SOPInstanceUID' in dicom_file else None

                if patient_uid and study_uid and series_uid and instance_uid:
                    dicom_data.append([file_path, input_folder, patient_uid, study_uid, series_uid, instance_uid])
            except Exception as e:
                # If the file is not a valid DICOM file, continue to the next file
                continue

            # Update dicomDf
            self.dicomDf = pd.DataFrame(dicom_data, columns=['Filepath', 'InputDirectory', 'PatientUID', 'StudyUID', 'SeriesUID', 'InstanceUID'])
            self.nextDicomDfIndex = 0

        # Close the progress dialog
        progress_dialog.setValue(total_files)
        progress_dialog.close()

        # Return the number of rows in the dataframe
        return len(self.dicomDf)

    def updateCurrentFrame(self):
        logging.info('updateCurrentFrame')

        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found, cannot update current frame.")
            return

        # Get the current frame index from the sequence browser
        currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())  # TODO: investigate whey this could be negative!
        logging.debug(f"Updating frame {currentFrameIndex} with {len(self.pleuraLines)} pleura lines and {len(self.bLines)} b-lines")

        # Check if annotations already has a list of frame annotations. Create it if it doesn't exist.
        if 'frame_annotations' not in self.annotations:
            self.annotations['frame_annotations'] = []

        # Find existing frame annotation for currentFrameIndex
        existing = next((f for f in self.annotations['frame_annotations']
                         if int(f.get("frame_number", -1)) == currentFrameIndex), None)
        if not existing:
            # create an empty frame and append it to annotations
            existing = {
                "frame_number": currentFrameIndex,
                "coordinate_space": "RAS",
                "pleura_lines": [],
                "b_lines": []
            }
            self.annotations['frame_annotations'].append(existing)

        # Remove all lines, we will add only the visible lines back in and any updates will take here
        existing['pleura_lines'] = []
        existing['b_lines'] = []

        pleura_saved = 0
        for i, markupNode in enumerate(self.pleuraLines):
            displayNode = markupNode.GetDisplayNode() if markupNode else None
            is_visible = displayNode.GetVisibility() if displayNode else False
            num_points = markupNode.GetNumberOfControlPoints() if markupNode else 0

            # Only save visible nodes with valid coordinates
            if not is_visible:
                continue  # Skip hidden nodes

            coordinates = []
            for j in range(num_points):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(j, coord)
                coordinates.append(coord)

            validation_json = markupNode.GetAttribute("validation")
            try:
                validation = json.loads(validation_json) if validation_json else {"status": "unadjudicated"}
            except json.JSONDecodeError:
                validation = {"status": "unadjudicated"}

            if coordinates and len(coordinates) >= 2:  # Only save lines with at least 2 points
                line_data = {
                    "rater": markupNode.GetAttribute("rater"),
                    "line": {"points": coordinates},
                    "validation": validation
                }
                existing['pleura_lines'].append(line_data)
                pleura_saved += 1

        bline_saved = 0
        for i, markupNode in enumerate(self.bLines):
            displayNode = markupNode.GetDisplayNode() if markupNode else None
            is_visible = displayNode.GetVisibility() if displayNode else False
            num_points = markupNode.GetNumberOfControlPoints() if markupNode else 0

            # Only save visible nodes with valid coordinates
            if not is_visible:
                continue  # Skip hidden nodes

            coordinates = []
            for j in range(num_points):
                coord = [0, 0, 0]
                markupNode.GetNthControlPointPosition(j, coord)
                coordinates.append(coord)

            validation_json = markupNode.GetAttribute("validation")
            try:
                validation = json.loads(validation_json) if validation_json else {"status": "unadjudicated"}
            except json.JSONDecodeError:
                validation = {"status": "unadjudicated"}

            if coordinates and len(coordinates) >= 2:  # Only save lines with at least 2 points
                line_data = {
                    "rater": markupNode.GetAttribute("rater"),
                    "line": {"points": coordinates},
                    "validation": validation
                }
                existing['b_lines'].append(line_data)
                bline_saved += 1

        # Update the markups in the scene to match the annotation data
        self.updateLineMarkups()

        # At the end of updateCurrentFrame, always update overlays
        self.updateOverlayVolume()


    def removeFrame(self, frameIndex):
        logging.info(f"removeFrame -- frameIndex: {frameIndex}")
        if self.annotations is None:
            logging.warning("removeFrame (adjudicate): No annotations loaded")
            return

        # Remove the frame index from the list of frame annotations
        self.annotations["frame_annotations"] = [
            fa for fa in self.annotations.get("frame_annotations", [])
            if int(fa.get("frame_number", -1)) != frameIndex
        ]


    def loadPreviousSequence(self):
        if self.dicomDf is None:
            return None

        if self.nextDicomDfIndex <= 1:
            return None
        else:
            self.nextDicomDfIndex -= 2
            return self.loadNextSequence()

    def clearScene(self):
        slicer.mrmlScene.Clear(0)
        self.annotations = None
        self.pleuraLines = []
        self.bLines = []
        self.sequenceBrowserNode = None
        # Reset overlay volume reference in parameter node
        if self.parameterNode:
            self.parameterNode.overlayVolume = None

    def convert_lps_to_ras(self, annotations: list):
        for frame in annotations:
            if frame.get("coordinate_space", "RAS") == "LPS":
                for line_group in ["pleura_lines", "b_lines"]:
                    for entry in frame.get(line_group, []):
                        points = entry["line"]["points"]
                        for point in points:
                            point[0] = -point[0]  # Negate X (Left → Right)
                            point[1] = -point[1]  # Negate Y (Posterior → Anterior)
                frame["coordinate_space"] = "RAS"  # Update coordinate_space

    # Use deepcopy because in-memory data should not be changed to LPS
    def convert_ras_to_lps(self, annotated_frames: list):
        # deepcopy so that modifications do not affect self.annotations
        save_data = copy.deepcopy(self.annotations)
        # deepcopy so that changes to the annotated frames don't actually affect the frames passed in
        copied_frames = copy.deepcopy(annotated_frames)
        for frame in copied_frames:
            if frame.get("coordinate_space", "RAS") == "RAS":
                for line_group in ["pleura_lines", "b_lines"]:
                    for entry in frame.get(line_group, []):
                        points = entry["line"]["points"]
                        for point in points:
                            point[0] = -point[0]  # Negate X (Right → Left)
                            point[1] = -point[1]  # Negate Y (Anterior → Posterior)
                frame["coordinate_space"] = "LPS"  # Update coordinate_space
        save_data['frame_annotations'] = copied_frames
        return save_data # a copy of the data, so caller has to save

    def _updateMarkupsAndOverlayProgrammatically(self, parameterNode=None):
        if parameterNode is None:
            parameterNode = self.getParameterNode()
        self._isProgrammaticUpdate = True
        try:
            self.updateLineMarkups()
            ratio = self.updateOverlayVolume()
            if ratio is not None:
                parameterNode.pleuraPercentage = ratio * 100
            else:
                parameterNode.pleuraPercentage = 0.0
        finally:
            self._isProgrammaticUpdate = False

    def loadNextSequence(self):
        """
        Load the next sequence in the dataframe.
        Returns the index of the loaded sequence in the dataframe or None if no more sequences are available.
        """
        # Save current depth guide mode
        currentDepthGuideMode = self.depthGuideMode
        parameterNode = self.getParameterNode()
        if not parameterNode:
            logging.error("No parameter node found, cannot load next sequence.")
            return None

        # Clear the scene
        self.clearScene()

        if self.dicomDf is None:
            parameterNode.dfLoaded = False
            return None
        else:
            parameterNode.dfLoaded = True

        if self.nextDicomDfIndex >= len(self.dicomDf):
            return None

        nextDicomFilepath = self.dicomDf.iloc[self.nextDicomDfIndex]['Filepath']
        dir_path = self.dicomDf.iloc[self.nextDicomDfIndex]['InputDirectory']
        # increment the index to the next DICOM file
        self.nextDicomDfIndex += 1
        base_name = os.path.splitext(os.path.basename(nextDicomFilepath))[0]

        # Make sure a temporary folder for the DICOM files exists
        tempDicomDir = slicer.app.temporaryPath + '/AnonymizeUltrasound'
        if not os.path.exists(tempDicomDir):
            os.makedirs(tempDicomDir)

        # Delete all files in the temporary folder
        for file in os.listdir(tempDicomDir):
            os.remove(os.path.join(tempDicomDir, file))

        # Clear markup cache to force update on reload
        self._lastMarkupFrameIndex = None
        self._lastMarkupFrameHash = None

        # Copy DICOM file to temporary folder
        shutil.copy(nextDicomFilepath, tempDicomDir)

        loadedNodeIDs = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(tempDicomDir, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

        logging.info(f"Loaded {len(loadedNodeIDs)} nodes")

        # Check loadedNodeIDs and collect sequence browser nodes to display them later
        currentSequenceBrowser = None
        for nodeID in loadedNodeIDs:
            currentSequenceBrowser = slicer.mrmlScene.GetNodeByID(nodeID)
            if currentSequenceBrowser is not None and currentSequenceBrowser.IsA("vtkMRMLSequenceBrowserNode"):
                self.currentDicomHeader = self.dicomHeaderDictForBrowserNode(currentSequenceBrowser)
                if self.currentDicomHeader is None:
                    logging.error(f"Could not find DICOM header for sequence browser node {currentSequenceBrowser.GetID()}")
                break

        # Get the current proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        inputUltrasoundNode = currentSequenceBrowser.GetProxyNode(masterSequenceNode)

        # Make sure the proxy node is a volume node and save it for later
        if inputUltrasoundNode is not None:
            if not (inputUltrasoundNode.IsA("vtkMRMLScalarVolumeNode") or inputUltrasoundNode.IsA("vtkMRMLVectorVolumeNode")):
                logging.error(f"Proxy node is not a volume node")
                return None

        previousNodeState = parameterNode.StartModify()

        self.sequenceBrowserNode = currentSequenceBrowser
        self.sequenceBrowserNode.SetPlaybackLooped(False)
        parameterNode.inputVolume = inputUltrasoundNode
        parameterNode.unsavedChanges = False  # Reset unsaved changes when loading new sequence

        # Restore depth guide mode
        self.depthGuideMode = currentDepthGuideMode
        logging.debug(f"Restored depthGuideMode to {self.depthGuideMode} after loading sequence")

        ultrasoundArray = slicer.util.arrayFromVolume(inputUltrasoundNode)
        # Mask array should be the same size as the ultrasound array, but with 3 channels
        maskArray = np.zeros([1, ultrasoundArray.shape[1], ultrasoundArray.shape[2], 3], dtype=np.uint8)

        # Initialize the mask volume to be the same size as the ultrasound volume but with all voxels set to 0
        overlayVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", "Overlay")
        overlayVolume.SetSpacing(inputUltrasoundNode.GetSpacing())
        overlayVolume.SetOrigin(inputUltrasoundNode.GetOrigin())
        ijkToRas = vtk.vtkMatrix4x4()
        inputUltrasoundNode.GetIJKToRASMatrix(ijkToRas)
        overlayVolume.SetIJKToRASMatrix(ijkToRas)
        overlayImageData = vtk.vtkImageData()
        overlayImageData.SetDimensions(ultrasoundArray.shape[1], ultrasoundArray.shape[2], 1)
        overlayImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        overlayVolume.SetAndObserveImageData(overlayImageData)
        # overlayVolume.CreateDefaultDisplayNodes()
        slicer.util.updateVolumeFromArray(overlayVolume, maskArray)
        parameterNode.overlayVolume = overlayVolume

        # Load all annotations with the same base prefix and deeply merge frame_annotations by frame_number
        self.seenRaters = []
        merged_data = {}
        merged_data["frame_annotations"] = []
        adjudication_path = os.path.join(dir_path, f"{base_name}.adjudication.json")
        if os.path.exists(adjudication_path):
            try:
                with open(adjudication_path, 'r') as f:
                    adjudication = json.load(f)
                    self.convert_lps_to_ras(adjudication.get("frame_annotations", []))
                    # just use the data we read as it has all the lines and other data we need
                    merged_data = adjudication
            except Exception as e:
                logging.warning(f"Failed to load adjudication file {adjudication_path}: {e}")
        else:
            filepaths = glob.glob(f"{dir_path}/**/{base_name}.json", recursive=True) + glob.glob(f"{dir_path}/**/{base_name}.*.json", recursive=True)
            for filepath in filepaths:
                try:
                    with open(filepath, 'r') as f:
                        ann = json.load(f)
                        self.convert_lps_to_ras(ann.get("frame_annotations", []))
                        # Merge non-frame_annotations keys with conflict check
                        for k, v in ann.items():
                            if k == "frame_annotations":
                                continue
                            if k not in merged_data:
                                merged_data[k] = v
                            elif merged_data[k] in [None, "", [], {}]:
                                merged_data[k] = v
                            elif isinstance(merged_data[k], list) and isinstance(v, list):
                                merged_data[k].extend(v)
                                merged_data[k] = list(dict.fromkeys(merged_data[k]))
                            elif merged_data[k] != v:
                                logging.warning(f"Conflicting values for key '{k}': {merged_data[k]} vs {v}, keeping first value")

                        # Only merge pleura_lines and b_lines for frames with matching frame_number
                        for frame in ann.get("frame_annotations", []):
                            frame_number = frame["frame_number"]
                            matched = next((f for f in merged_data["frame_annotations"] if f["frame_number"] == frame_number), None)
                            if matched:
                                matched["pleura_lines"].extend(frame.get("pleura_lines", []))
                                matched["b_lines"].extend(frame.get("b_lines", []))
                            else:
                                merged_data["frame_annotations"].append({
                                    "frame_number": frame["frame_number"],
                                    "coordinate_space": frame.get("coordinate_space", "RAS"),
                                    "pleura_lines": frame.get("pleura_lines", []),
                                    "b_lines": frame.get("b_lines", []),
                                    "validation": frame.get("validation", "unadjudated")
                                })
                except Exception as e:
                    logging.warning(f"Failed to load annotation file {filepath}: {e}")

        self.annotations = merged_data

        # Extract and set up raters using the centralized method
        self.extractAndSetupRaters()
        # Clean up duplicates from the loaded annotation data
        self.cleanupAnnotationDuplicates()

        # Initialize markup nodes based on loaded annotations
        self.initializeMarkupNodesFromAnnotations()
        current_rater = self.getParameterNode().rater.strip().lower()
        if current_rater in self.seenRaters:
            self.seenRaters.remove(current_rater)
        if "__selected_node__" in self.seenRaters:
            self.seenRaters.remove("__selected_node__")
        if "__adjudicated_node__" in self.seenRaters:
            self.seenRaters.remove("__adjudicated_node__")
        # put current rater at the top
        self.seenRaters = [current_rater, "__selected_node__", "__adjudicated_node__"] + sorted(self.seenRaters)
        self.realRaters = [r for r in self.seenRaters if r != "__selected_node__" and r != "__adjudicated_node__"]
        self.setSelectedRaters(self.realRaters)

        # Set programmatic update flag to prevent unsavedChanges from being set
        self._updateMarkupsAndOverlayProgrammatically(parameterNode)
        parameterNode.EndModify(previousNodeState)

        # Set overlay volume as foreground in slice viewers
        redSliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceCompositeNode()
        redSliceCompositeNode.SetForegroundVolumeID(overlayVolume.GetID())
        redSliceCompositeNode.SetForegroundOpacity(0.12)
        redSliceCompositeNode.SetCompositing(2)

        displayNode = overlayVolume.GetDisplayNode()
        displayNode.SetWindow(255)
        displayNode.SetLevel(127)

        # Observe the ultrasound image for changes
        self.addObserver(self.sequenceBrowserNode, vtk.vtkCommand.ModifiedEvent, self.onSequenceBrowserModified)

        # Return the index of the loaded sequence in the dataframe
        return self.nextDicomDfIndex

    def onSequenceBrowserModified(self, caller, event):
        self._updateMarkupsAndOverlayProgrammatically(None)

    def createMarkupLine(self, name, rater, coordinates, color=[1, 1, 0], validation=None):
        markupNode = super().createMarkupLine(name, rater, coordinates, color)
        if markupNode is not None and validation is not None:
            markupNode.SetAttribute("validation", json.dumps(validation))

        return markupNode

    def _updateMarkupNode(self, node, entry, selectedNodeID):
        # Check if node is still valid
        if not node or not slicer.mrmlScene.IsNodePresent(node):
            return

        coordinates = entry.get("line", {}).get("points", [])
        rater = entry.get("rater", "")
        validation = entry.get("validation", None)
        if not validation or not validation.get("status"):
            validation = {"status": "unadjudicated"}
        node.SetAttribute("rater", rater)
        node.SetAttribute("validation", json.dumps(validation))
        status = validation.get("status", "unadjudicated")

        # Ensure display node exists
        displayNode = node.GetDisplayNode()
        if displayNode is None:
            # Check if scene is valid before creating display nodes
            if not slicer.mrmlScene:
                return
            try:
                node.CreateDefaultDisplayNodes()
                displayNode = node.GetDisplayNode()
                if displayNode is None:
                    logging.error(f"Failed to create display node for markup node {node.GetName()}")
                    return
            except Exception as e:
                logging.error(f"Exception creating display node for {node.GetName()}: {e}")
                return

        # Check if this node is selected - apply selected highlighting regardless of validation status
        isSelected = (node.GetID() == selectedNodeID)

        if isSelected:
            # Apply selected highlighting using the __selected_node__ color
            _, selected_node_color = self.getColorsForRater("__selected_node__")
            displayNode.SetSelectedColor(selected_node_color)
        else:
            # Apply colors based on mode and validation status
            if status == "unadjudicated":
                # In adjudicator mode, unadjudicated lines use current rater's colors
                current_rater = self.getCurrentRater()
                if node in self.pleuraLines:
                    color_pleura, _ = self.getColorsForRater(current_rater)
                    displayNode.SetSelectedColor(color_pleura)
                else:
                    _, color_bline = self.getColorsForRater(current_rater)
                    displayNode.SetSelectedColor(color_bline)
            else:
                # In adjudicator mode, validated/invalidated/duplicated lines use __adjudicated_node__ color
                if node in self.pleuraLines:
                    color_pleura, _ = self.getColorsForRater("__adjudicated_node__")
                    displayNode.SetSelectedColor(color_pleura)
                else:
                    _, color_bline = self.getColorsForRater("__adjudicated_node__")
                    displayNode.SetSelectedColor(color_bline)
        # Set visibility, opacity, and line style based on validation status and adjudicator mode
        status = validation.get("status", "unadjudicated")
        displayNode.SetGlyphTypeFromString("Circle2D")
        displayNode.SetGlyphScale(2.0)
        displayNode.SetLineThickness(0.25)
        if status == "invalidated":
            showInvalidAndDuplicate = self.parameterNode.showInvalidAndDuplicate if self.parameterNode else True
            displayNode.SetVisibility(showInvalidAndDuplicate)
            displayNode.SetGlyphTypeFromString("Cross2D")
            displayNode.SetOpacity(0.40)
        elif status == "duplicated":
            showInvalidAndDuplicate = self.parameterNode.showInvalidAndDuplicate if self.parameterNode else True
            displayNode.SetVisibility(showInvalidAndDuplicate)
            displayNode.SetGlyphTypeFromString("Diamond2D")
            displayNode.SetOpacity(0.50)
        elif status == "unadjudicated":
            displayNode.SetVisibility(True)
            if not isSelected:
                displayNode.SetOpacity(0.65)
        else:  # validated
            displayNode.SetVisibility(True)
            if not isSelected:
                displayNode.SetOpacity(0.85)

        # Selected node highlighting is independent of validation status
        if isSelected:
            displayNode.SetGlyphScale(3.0)
            displayNode.SetOpacity(1.0)

        node.RemoveAllControlPoints()
        for pt in coordinates:
            node.AddControlPointWorld(*pt)

    def _updateMarkupNodesForFrame(self, frame):
        """
        Update markup nodes for pleura and b-lines for the given frame.
        """

        # Check if scene is valid before proceeding
        if not slicer.mrmlScene:
            return

        pleura_entries = [entry for entry in frame.get("pleura_lines", []) if entry.get("rater") in self.selectedRaters]
        bline_entries = [entry for entry in frame.get("b_lines", []) if entry.get("rater") in self.selectedRaters]

        # Get currently selected node ID for highlighting
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectedNodeID = selectionNode.GetActivePlaceNodeID() if selectionNode else None

        # Ensure we have enough markup nodes (create more if needed, but don't remove for performance)
        while len(self.pleuraLines) < len(pleura_entries):
            self.pleuraLines.append(self.createMarkupLine("Pleura", "", [], [1,1,0]))
        while len(self.bLines) < len(bline_entries):
            self.bLines.append(self.createMarkupLine("B-line", "", [], [0,1,1]))

        # Update pleura markups
        for i, entry in enumerate(pleura_entries):
            node = self.pleuraLines[i]
            self._updateMarkupNode(node, entry, selectedNodeID)

        # Hide unused pleura markups
        for i in range(len(pleura_entries), len(self.pleuraLines)):
            displayNode = self.pleuraLines[i].GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(False)

        # Hide pleura markups for non-selected raters
        for node in self.pleuraLines:
            nodeRater = node.GetAttribute("rater")
            if nodeRater not in self.selectedRaters:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)

        # Update b-line markups
        for i, entry in enumerate(bline_entries):
            node = self.bLines[i]
            self._updateMarkupNode(node, entry, selectedNodeID)

        # Hide unused b-line markups
        for i in range(len(bline_entries), len(self.bLines)):
            displayNode = self.bLines[i].GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(False)

       # Hide b-line markups for non-selected raters
        for node in self.bLines:
            nodeRater = node.GetAttribute("rater")
            if nodeRater not in self.selectedRaters:
                displayNode = node.GetDisplayNode()
                if displayNode:
                    displayNode.SetVisibility(False)

    def updateLineMarkups(self):
        """
        Update the line markups to match the annotations at the current frame index. Reuse markups instead of deleting/creating them every frame. Only update if frame or annotation data has changed.
        """
        # Check if scene is valid before proceeding
        if not slicer.mrmlScene:
            return

        # Initialize cache attributes if not present
        if not hasattr(self, '_lastMarkupFrameIndex'):
            self._lastMarkupFrameIndex = None
        if not hasattr(self, '_lastMarkupFrameHash'):
            self._lastMarkupFrameHash = None

        if self.annotations is None:
            # this is legit when we are loading a new sequence as we change selections when loading a new sequence
            # which causes updateLineMarkups to be called before the annotations are loaded
            logging.debug("updateLineMarkups (adjudicate): No annotations loaded")
            return

        if self.sequenceBrowserNode is None:
            logging.warning("No sequence browser node found")
            return

        currentFrameIndex = max(0, self.sequenceBrowserNode.GetSelectedItemNumber())

        if 'frame_annotations' not in self.annotations:
            logging.debug("No frame annotations found")
            return

        frame = next((item for item in self.annotations['frame_annotations'] if str(item.get("frame_number")) == str(currentFrameIndex)), None)
        # Compute a hash of the frame annotation data and selected raters for caching
        frame_hash = None
        if frame is not None:
            try:
                selectionNode = slicer.app.applicationLogic().GetSelectionNode()
                selectedNodeID = selectionNode.GetActivePlaceNodeID() if selectionNode else None
                # Include selected raters and selected node in the hash so selection changes trigger updates
                cache_data = {
                    'frame': frame,
                    'selectedRaters': sorted(list(self.selectedRaters)) if hasattr(self, 'selectedRaters') else [],
                    'selectedNodeID': selectedNodeID
                }
                frame_hash = hash(json.dumps(cache_data, sort_keys=True))
            except Exception:
                frame_hash = None

        # Early return if frame index, annotation data, and selected raters are unchanged
        if self._lastMarkupFrameIndex == currentFrameIndex and self._lastMarkupFrameHash == frame_hash:
            return

        # Update cache after check
        self._lastMarkupFrameIndex = currentFrameIndex
        self._lastMarkupFrameHash = frame_hash

        # Set programmatic update flag to prevent unsavedChanges from being set
        self._isProgrammaticUpdate = True

        # Batch scene updates using StartState/EndState
        slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
        try:
            if frame is None:
                # Hide all markups if no frame data
                for node in self.pleuraLines:
                    displayNode = node.GetDisplayNode()
                    if displayNode:
                        displayNode.SetVisibility(False)
                for node in self.bLines:
                    displayNode = node.GetDisplayNode()
                    if displayNode:
                        displayNode.SetVisibility(False)
                return

            self._updateMarkupNodesForFrame(frame)
        finally:
            slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)
            # Reset programmatic update flag
            self._isProgrammaticUpdate = False

    def updateOverlayVolume(self):
        """
        Update the overlay volume based on the annotations.

        :return: The ratio of green pixels to blue pixels in the overlay volume. None if inputs not defined yet.
        """
        parameterNode = self.getParameterNode()

        if parameterNode.overlayVolume is None:
            logging.debug("updateOverlayVolume: No overlay volume found! Cannot update overlay volume.")
            return None

        if self.annotations is None:
            logging.warning("updateOverlayVolume (adjudicate): No annotations loaded")
            # Make sure all voxels are set to 0
            parameterNode.overlayVolume.GetImageData().GetPointData().GetScalars().Fill(0)
            return None

        if parameterNode.inputVolume is None:
            logging.debug("No input volume found, not updating overlay volume.")
            return None

        # if we are using multiple raters and have selected more than one, don't show overlay volume
        if hasattr(self, "selectedRaters") and len(self.selectedRaters) > 1:
            overlayArray = slicer.util.arrayFromVolume(parameterNode.overlayVolume)
            overlayArray[:] = 0
            overlayArray = self._applyDepthGuideToMask(overlayArray, parameterNode)
            slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, overlayArray)
            slicer.util.showStatusMessage("Overlay hidden: multiple raters selected", 3000)
            return None

        if self.annotations is None:
            logging.warning("updateOverlayVolume (adjudicate): No annotations loaded")
            # Make sure all voxels are set to 0
            parameterNode.overlayVolume.GetImageData().GetPointData().GetScalars().Fill(0)
            return None

        # If no raters are selected, do not draw any mask
        if hasattr(self, "selectedRaters") and not self.selectedRaters:
            overlayArray = slicer.util.arrayFromVolume(parameterNode.overlayVolume)
            overlayArray[:] = 0
            overlayArray = self._applyDepthGuideToMask(overlayArray, parameterNode)
            slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, overlayArray)
            slicer.util.showStatusMessage("Overlay hidden: no raters selected", 3000)
            return None

        if parameterNode.inputVolume is None:
            logging.debug("No input volume found, not updating overlay volume.")
            return None

        ultrasoundArray = slicer.util.arrayFromVolume(parameterNode.inputVolume)

        # Mask array should be the same size as the ultrasound array
        # Make the mask array RGB color regardless of the number of channels in the ultrasound array
        maskArray = np.zeros([1, ultrasoundArray.shape[1], ultrasoundArray.shape[2], 3], dtype=np.uint8)
        ijkToRas = vtk.vtkMatrix4x4()
        parameterNode.inputVolume.GetIJKToRASMatrix(ijkToRas)
        rasToIjk = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)

        # Add pleura lines to mask array using full RGB overlay
        for markupNode in self.pleuraLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else None
            validation_json = markupNode.GetAttribute("validation")
            status = None
            if validation_json:
                try:
                    validation = json.loads(validation_json)
                    status = validation.get("status", None)
                except Exception:
                    status = None
            if status != "validated":
                continue

            if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue

            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1 = [0, 0, 0]
                coord2 = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
                coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
                coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
                coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
                # Draw mask fan between coord1 and coord2
                sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2, value=255)
                # Add sectorArray to maskArray by maximum compounding
                maskArray[0, :, :, 2] = np.maximum(maskArray[0, :, :, 2], sectorArray)

        # Add B-lines to mask array using full RGB overlay
        for markupNode in self.bLines:
            nodeRater = markupNode.GetAttribute("rater") if markupNode else None
            validation_json = markupNode.GetAttribute("validation")
            status = None
            if validation_json:
                try:
                    validation = json.loads(validation_json)
                    status = validation.get("status", None)
                except Exception:
                    status = None
            if status != "validated":
                continue

            if hasattr(self, "selectedRaters") and self.selectedRaters and nodeRater not in self.selectedRaters:
                continue
            if not markupNode.GetDisplayNode().GetVisibility():
                continue

            for i in range(markupNode.GetNumberOfControlPoints() - 1):
                coord1 = [0, 0, 0]
                coord2 = [0, 0, 0]
                markupNode.GetNthControlPointPosition(i, coord1)
                markupNode.GetNthControlPointPosition(i + 1, coord2)
                coord1 = rasToIjk.MultiplyPoint(coord1 + [1])
                coord2 = rasToIjk.MultiplyPoint(coord2 + [1])
                coord1 = [int(round(coord1[0])), int(round(coord1[1])), int(round(coord1[2]))]
                coord2 = [int(round(coord2[0])), int(round(coord2[1])), int(round(coord2[2]))]
                # Draw mask fan between coord1 and coord2
                sectorArray = self.createSectorMaskBetweenPoints(ultrasoundArray, coord1, coord2)
                # Add sectorArray to maskArray by maximum compounding
                maskArray[0, :, :, 1] = np.maximum(maskArray[0, :, :, 1], sectorArray)

        # Erase all B-lines pixels where there is no pleura line
        maskArray[0, :, :, 1] = np.where(maskArray[0, :, :, 2] == 0, 0, maskArray[0, :, :, 1])

        # Calculate the amount of blue pixels in maskArray and green pixels in maskArray
        bluePixels = np.count_nonzero(maskArray[0, :, :, 2])
        greenPixels = np.count_nonzero(maskArray[0, :, :, 1])

        # apply depthGuide if enabled
        maskArray = self._applyDepthGuideToMask(maskArray, parameterNode)

        # Compute a hash of the mask array for fast comparison
        mask_hash = zlib.crc32(maskArray.tobytes())
        if hasattr(self, '_lastOverlayMaskHash') and self._lastOverlayMaskHash == mask_hash:
            # No change, skip update
            if bluePixels == 0:
                parameterNode.pleuraPercentage = 0.0
                return 0.0
            else:
                parameterNode.pleuraPercentage = greenPixels / bluePixels * 100
                return greenPixels / bluePixels
        # Update the overlay volume
        slicer.util.updateVolumeFromArray(parameterNode.overlayVolume, maskArray)
        self._lastOverlayMaskHash = mask_hash

        # Return the ratio of green pixels to blue pixels
        if bluePixels == 0:
            parameterNode.pleuraPercentage = 0.0
            return 0.0
        else:
            parameterNode.pleuraPercentage = greenPixels / bluePixels * 100
            return greenPixels / bluePixels

    def _resetMarkupCache(self):
        """
        Reset the markup cache to force immediate updates when validation status changes.
        """
        if hasattr(self, '_lastMarkupFrameIndex'):
            self._lastMarkupFrameIndex = None
        if hasattr(self, '_lastMarkupFrameHash'):
            self._lastMarkupFrameHash = None

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def extractAndSetupRaters(self):
        """
        Extract all unique raters from the loaded annotations and set up selectedRaters.
        This centralizes the rater extraction logic to avoid duplication.
        """
        # Extract all unique raters from the loaded annotations
        seenRaters = []
        if self.annotations and 'frame_annotations' in self.annotations:
            for frame in self.annotations['frame_annotations']:
                for line_type in ['pleura_lines', 'b_lines']:
                    for line in frame.get(line_type, []):
                        rater = line.get('rater')
                        if rater and rater not in seenRaters:
                            seenRaters.append(rater)

        parameterNode = self.getParameterNode()
        current_rater = parameterNode.rater.strip().lower()
        if current_rater in seenRaters:
            seenRaters.remove(current_rater)
        # Remove __selected_node__ if it exists (to avoid duplicates)
        if "__selected_node__" in seenRaters:
            seenRaters.remove("__selected_node__")
        # Remove __adjudicated_node__ if it exists (to avoid duplicates)
        if "__adjudicated_node__" in seenRaters:
            seenRaters.remove("__adjudicated_node__")
        # Remove current_rater if it exists (to avoid duplicates)
        if current_rater in seenRaters:
            seenRaters.remove(current_rater)
        # Now build the list: current_rater, __selected_node__, __adjudicated_node__ then sorted rest
        # we need to add __selected_node__, __adjudicated_node__ to the list to ensure that the selected line is always visible and
        # uses a different color than the other lines
        self.seenRaters = [current_rater, "__selected_node__", "__adjudicated_node__"] + sorted(seenRaters)
        # Select all real raters by default (exclude __selected_node__)
        self.realRaters = [r for r in self.seenRaters if r != "__selected_node__" and r != "__adjudicated_node__"]
        self.setSelectedRaters(set(self.realRaters))

    def cleanupAnnotationDuplicates(self):
        """
        Remove duplicate lines from the annotation data in memory.
        Lines are only considered duplicates if they have identical points AND the same rater.
        This prevents duplicates from being displayed. We shouldn't need this anymore but it's here just in case
        there are files that have duplicates that got saved from previous versions of the module.
        """
        if not self.annotations or 'frame_annotations' not in self.annotations:
            return

        total_removed = 0
        has_duplicates = False

        for frame in self.annotations['frame_annotations']:
            frame_num = frame.get('frame_number', 'unknown')

            # Check pleura lines for duplicates
            if 'pleura_lines' in frame:
                original_count = len(frame['pleura_lines'])
                seen_pleura = set()
                unique_pleura = []

                for i, entry in enumerate(frame['pleura_lines']):
                    points = entry.get('line', {}).get('points', [])
                    rater = entry.get('rater', '')
                    # Create a hash of points and rater
                    points_hash = hash(tuple(tuple(pt) for pt in points) + (rater,))

                    if points_hash not in seen_pleura:
                        seen_pleura.add(points_hash)
                        unique_pleura.append(entry)
                    else:
                        has_duplicates = True

                if has_duplicates:
                    frame['pleura_lines'] = unique_pleura
                    removed = original_count - len(unique_pleura)
                    total_removed += removed

            # Check b-lines for duplicates
            if 'b_lines' in frame:
                original_count = len(frame['b_lines'])
                seen_blines = set()
                unique_blines = []

                for i, entry in enumerate(frame['b_lines']):
                    points = entry.get('line', {}).get('points', [])
                    rater = entry.get('rater', '')
                    # Create a hash of points and rater
                    points_hash = hash(tuple(tuple(pt) for pt in points) + (rater,))

                    if points_hash not in seen_blines:
                        seen_blines.add(points_hash)
                        unique_blines.append(entry)
                    else:
                        has_duplicates = True

                if has_duplicates:
                    frame['b_lines'] = unique_blines
                    removed = original_count - len(unique_blines)
                    total_removed += removed

        if has_duplicates:
            # Reinitialize markup nodes if needed after cleanup
            self.reinitializeMarkupNodesIfNeeded()

    def initializeMarkupNodesFromAnnotations(self):
        """
        Initialize markup nodes based on the maximum number needed across all frames.
        This ensures we start with the right number of nodes and don't need to create/remove them constantly.
        """
        if not self.annotations or 'frame_annotations' not in self.annotations:
            return

        # Calculate maximum number of lines needed across all frames
        max_pleura_lines = 0
        max_blines = 0

        for frame in self.annotations['frame_annotations']:
            pleura_count = len(frame.get('pleura_lines', []))
            bline_count = len(frame.get('b_lines', []))
            max_pleura_lines = max(max_pleura_lines, pleura_count)
            max_blines = max(max_blines, bline_count)

        # Add some buffer for new lines (at least 2 of each type, but cap at reasonable limits)
        max_pleura_lines = max(max_pleura_lines, 2)
        max_blines = max(max_blines, 2)

        # Cap the maximum to prevent excessive node creation
        max_pleura_lines = min(max_pleura_lines, 10)  # Cap at 10 pleura nodes
        max_blines = min(max_blines, 10)              # Cap at 10 B-line nodes

        # Clear existing nodes
        self.clearSceneLines()

        # Create the required number of nodes
        for i in range(max_pleura_lines):
            self.pleuraLines.append(self.createMarkupLine("Pleura", "", [], [1,1,0]))
        for i in range(max_blines):
            self.bLines.append(self.createMarkupLine("B-line", "", [], [0,1,1]))

    def reinitializeMarkupNodesIfNeeded(self):
        """
        Reinitialize markup nodes if the current number doesn't match what's needed.
        This is useful after cleaning up duplicates or when the annotation data changes significantly.
        """
        if not self.annotations or 'frame_annotations' not in self.annotations:
            return

        # Calculate what we actually need now
        max_pleura_lines = 0
        max_blines = 0

        for frame in self.annotations['frame_annotations']:
            pleura_count = len(frame.get('pleura_lines', []))
            bline_count = len(frame.get('b_lines', []))
            max_pleura_lines = max(max_pleura_lines, pleura_count)
            max_blines = max(max_blines, bline_count)

        # Add buffer
        max_pleura_lines = max(max_pleura_lines, 2)
        max_blines = max(max_blines, 2)

        # Cap the maximum
        max_pleura_lines = min(max_pleura_lines, 10)
        max_blines = min(max_blines, 10)

        current_pleura = len(self.pleuraLines)
        current_blines = len(self.bLines)

        # Only reinitialize if there's a significant difference
        if (abs(current_pleura - max_pleura_lines) > 2 or
            abs(current_blines - max_blines) > 2):
            self.initializeMarkupNodesFromAnnotations()


#
# Register the module
#

if __name__ == "__main__":
    import sys
    import os
    import slicer

    # Add the module path to sys.path
    modulePath = os.path.dirname(os.path.abspath(__file__))
    if modulePath not in sys.path:
        sys.path.insert(0, modulePath)

    # Register the module
    import AdjudicateUltrasound
    slicer.modules.adjudicateultrasound = AdjudicateUltrasound.AdjudicateUltrasound(slicer.qSlicerApplication().moduleManager())
