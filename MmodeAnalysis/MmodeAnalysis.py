import cv2
import logging
import numpy as np
import os
from scipy.ndimage import map_coordinates, zoom
from typing import Annotated, Optional

import qt
import vtk
import pydicom
import pandas as pd
from datetime import datetime

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLMarkupsLineNode


#
# MmodeAnalysis
#


class MmodeAnalysis(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("MmodeAnalysis")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MmodeAnalysis">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # MmodeAnalysis1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MmodeAnalysis",
        sampleName="MmodeAnalysis1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "MmodeAnalysis1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="MmodeAnalysis1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="MmodeAnalysis1",
    )

    # MmodeAnalysis2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MmodeAnalysis",
        sampleName="MmodeAnalysis2",
        thumbnailFileName=os.path.join(iconsPath, "MmodeAnalysis2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="MmodeAnalysis2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="MmodeAnalysis2",
    )


#
# MmodeAnalysisParameterNode
#


@parameterNodeWrapper
class MmodeAnalysisParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    scanlineMarkup - Markups line node to define the scanline for M-mode conversion.
    measurementMarkup - Distance measure for the M-mode image.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    mmodeVolume: vtkMRMLScalarVolumeNode
    scanlineMarkup: vtkMRMLMarkupsLineNode
    measurementMarkup: vtkMRMLMarkupsLineNode


#
# MmodeAnalysisWidget
#


class MmodeAnalysisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    
    LAYOUT_VERTICAL2 = 542

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MmodeAnalysis.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MmodeAnalysisLogic()
        self.logic.setupScene()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        
        # If AnonymizeUltrasound/OutputDirectory setting is not found, then use the default value. Otherwise set directory on button widget.
        settings = qt.QSettings()
        defaultOutputDirectory = os.path.expanduser("~")
        outputDirectory = settings.value('MmodeAnalysis/OutputDirectory', defaultOutputDirectory)
        self.ui.outputDirectoryButton.directory = outputDirectory
        self.ui.outputDirectoryButton.directoryChanged.connect(self.updateOutputDirectoryFromWidget)
        
        self.ui.scanlineButton.connect("clicked(bool)", self.onScanlineButton)
        self.ui.measurementButton.connect("clicked(bool)", self.onMeasurementButton)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.saveButton.connect("clicked(bool)", self.onSaveButton)
        
        # Add custom layout
        layoutVertical2 = \
            """
            <layout type="vertical" split="true">
                <item splitSize="500">
                <view class="vtkMRMLSliceNode" singletontag="Red">
                    <property name="orientation" action="default">Axial</property>
                    <property name="viewlabel" action="default">R</property>
                    <property name="viewcolor" action="default">#F34A33</property>
                </view>
                </item>
                <item splitSize="300">
                <view class="vtkMRMLSliceNode" singletontag="Green">
                    <property name="orientation" action="default">Axial</property>
                    <property name="viewlabel" action="default">R</property>
                    <property name="viewcolor" action="default">#4AF333</property>
                </view>
                </item>
            </layout>
            """

        layoutManager = slicer.app.layoutManager()
        if not layoutManager.layoutLogic().GetLayoutNode().SetLayoutDescription(self.LAYOUT_VERTICAL2, layoutVertical2):
            layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(self.LAYOUT_VERTICAL2, layoutVertical2)
        
        # Add button to layout selector toolbar for this custom layout
        viewToolBar = slicer.util.mainWindow().findChild("QToolBar", "ViewToolBar")
        layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
        layoutSwitchActionParent = layoutMenu  # use `layoutMenu` to add inside layout list, use `viewToolBar` to add next the standard layout list
        layoutSwitchAction = layoutSwitchActionParent.addAction("M-mode view") # add inside layout list
        layoutSwitchAction.setData(self.LAYOUT_VERTICAL2)
        layoutSwitchAction.setIcon(qt.QIcon(":Icons/Go.png"))
        layoutSwitchAction.setToolTip("M-mode view")
        
        layoutManager.setLayout(self.LAYOUT_VERTICAL2)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGui)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
        
        # Create a scalar volume node for the output volume if it does not exist
        if not self._parameterNode.mmodeVolume:
            outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "M-mode")
            self._parameterNode.mmodeVolume = outputVolume
        
        if not self._parameterNode.scanlineMarkup:
            scanlineMarkup = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Scanline")
            scanlineMarkup.CreateDefaultDisplayNodes()
            self._parameterNode.scanlineMarkup = scanlineMarkup
        
        if not self._parameterNode.measurementMarkup:
            measurementMarkup = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Measurement")
            measurementMarkup.CreateDefaultDisplayNodes()
            self._parameterNode.measurementMarkup = measurementMarkup

    def setParameterNode(self, inputParameterNode: Optional[MmodeAnalysisParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGui)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._updateGui)
            self._updateGui()
    
    def onScanlineButton(self) -> None:
        """Select the markup line for scanline definition. Put mouse mode in placement mode."""
        
        scanlineNode = self.logic.getScanlineNode()
        scanlineNode.RemoveAllControlPoints()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
        selectionNode.SetActivePlaceNodeID(scanlineNode.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SwitchToSinglePlaceMode()
    
    def onMeasurementButton(self) -> None:
        """Select the markup line for measurement. Put mouse mode in placement mode."""
        
        measurementNode = self.logic.getMeasurementNode()
        measurementNode.RemoveAllControlPoints()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
        selectionNode.SetActivePlaceNodeID(measurementNode.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SwitchToSinglePlaceMode()
    
    def _updateGui(self, caller=None, event=None) -> None:
        # if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
        #     self.ui.applyButton.toolTip = _("Compute output volume")
        #     self.ui.applyButton.enabled = True
        # else:
        #     self.ui.applyButton.toolTip = _("Select input and output volume nodes")
        #     self.ui.applyButton.enabled = False
        return None

    def updateOutputDirectoryFromWidget(self):
        """
        Called when user changes the directory in the directory browser.
        """
        # Save current directory in application settings, so it is remembered when the module is re-opened

        directory = self.ui.outputDirectoryButton.directory
        settings = qt.QSettings()
        settings.setValue('MmodeAnalysis/OutputDirectory', directory)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(),
                               self.ui.outputSelector.currentNode(),
                               )
        
        # Set axial as the orientation for the green viewer
        slicer.util.getNode('vtkMRMLSliceNodeGreen').SetOrientationToAxial()
        # Set the output volume as the background volume in the green viewer
        slicer.app.layoutManager().sliceWidget("Green").sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.ui.outputSelector.currentNode().GetID())
        # Center the green viewer on the output volume
        slicer.app.layoutManager().sliceWidget("Green").sliceLogic().FitSliceToAll()
    
    def onSaveButton(self) -> None:
        """
        Callback function for Save button.
        """
        if self._parameterNode.mmodeVolume is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Save output volume", "Output volume not available")
        
        outputDirectory = self.ui.outputDirectoryButton.directory
        defaultFilepath = os.path.join(outputDirectory, "MmodeUltrasound.png")
        filename = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Save output volume", defaultFilepath, "PNG (*.png)")
        if filename:
            messageBox = qt.QMessageBox(qt.QMessageBox.Information, "Save output volume", "Saving output volume...")
            messageBox.setModal(True)
            messageBox.show()
            slicer.app.processEvents()
            try:
                self.logic.saveAnnotatedMmodeImage(filename)
                self.logic.saveMeasurement(outputDirectory, imageFilename=filename)
                messageBox.close()
            except Exception as e:
                messageBox.setText(f"Failed to save output volume: {e}")
                messageBox.setStandardButtons(qt.QMessageBox.Ok)
        
#
# MmodeAnalysisLogic
#


class MmodeAnalysisLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MmodeAnalysisParameterNode(super().getParameterNode())
    
    def getScanlineNode(self):
        parameterNode = self.getParameterNode()
        scanlineNode = parameterNode.scanlineMarkup
        if not scanlineNode:
            scanlineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Scanline")
            scanlineNode.CreateDefaultDisplayNodes()
            parameterNode.scanlineMarkup = scanlineNode
        return scanlineNode
        
    def getMeasurementNode(self):
        parameterNode = self.getParameterNode()
        measurementNode = parameterNode.measurementMarkup
        if not measurementNode:
            measurementNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Measurement")
            measurementNode.CreateDefaultDisplayNodes()
            parameterNode.measurementMarkup = measurementNode
        return measurementNode
    
    def setupScene(self):
        """
        Make sure required nodes are present in the scene.
        """
    
    def getDicomDataForBrowser(self, sequenceBrowser):
        """
        Get the DICOM data for the sequence browser.
        :param sequenceBrowser: Sequence browser node
        :return: pydicom dataset
        """
        dcmFilepath = sequenceBrowser.GetAttribute("DicomFile")
        if dcmFilepath is None:
            return None
        
        dcm = pydicom.dcmread(dcmFilepath)
        return dcm
        
    def sample_line(self, image, point1, point2, num_points=100, average_channels=False):
        """
        Sample pixel values along a line in an image, with an option to average across channels.

        Parameters:
        - image: np.array with shape (rows, columns, channels).
        - point1: Tuple (x1, y1) defining the start of the line.
        - point2: Tuple (x2, y2) defining the end of the line.
        - num_points: Number of points to sample along the line.
        - average_channels: If True, average the values across all channels.

        Returns:
        - samples: Array of sampled pixel values along the line. If average_channels is True,
                returns a 1D array of averaged values; otherwise, a 2D array with shape (num_points, channels).
        """

        # Ensure image is in (rows, columns, channels) format, if there is a third dimension
        if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image = np.transpose(image, (1, 2, 0))
        rows, cols, channels = image.shape
        
        img_dtype = image.dtype

        # If the second coordinate of point2 is greater than the second coordinate of point1, swap the points
        if point2[1] > point1[1]:
            point1, point2 = point2, point1

        # Generate line coordinates
        x = np.linspace(point1[0], point2[0], num_points)
        y = np.linspace(point1[1], point2[1], num_points)

        # Combine x and y coordinates
        line_coords = np.vstack((y, x))  # map_coordinates expects (rows, cols)

        # Sample along the line
        if average_channels:
            # Initialize an array for averaged samples
            samples = np.zeros(num_points, dtype=img_dtype)
            temp_samples = np.zeros(num_points)
            for i in range(channels):
                channel_samples = map_coordinates(image[:, :, i], line_coords, order=1)
                temp_samples += channel_samples
            temp_samples /= channels  # Average across channels
            samples = np.round(temp_samples).astype(img_dtype)
        else:
            # Sample for each channel
            samples = np.zeros((num_points, channels), dtype=img_dtype)
            for i in range(channels):
                samples[:, i] = map_coordinates(image[:, :, i], line_coords, order=1).astype(img_dtype)

        return samples

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                paperSpeed_mm_per_px: float = 25.0
                ) -> None:
        """
        Finds the sequence that the inputVolume is the proxy node of. Samples the inputVolume along the scanline across all frames of the sequence and writes the result to the outputVolume.
        :param inputVolume: B-mode 2D ultrasound
        :param scanlineMarkup: markup node that contains the scanline
        :param outputVolume: M-mode ultrasound
        """

        if not inputVolume:
            raise ValueError("Input volume is invalid")
        
        if not outputVolume:
            raise ValueError("Output volume is invalid")
        
        parameterNode = self.getParameterNode()
        
        scanlineMarkup = parameterNode.scanlineMarkup
        
        if (scanlineMarkup.GetNumberOfControlPoints() < 2):
            raise ValueError("Scanline markup requires at least 2 control points")

        import time

        startTime = time.time()
        logging.info("M-mode conversion started")

        # Get the scanline points in homogeneous RAS coordinates.
        scanlinePoints_RAS = np.zeros((2, 4))
        for i in range(2):
            scanlineMarkup.GetNthControlPointPositionWorld(i, scanlinePoints_RAS[i, :3])
            scanlinePoints_RAS[i, 3] = 1

        # Get the rasToIjk matrix
        rasToIjk = vtk.vtkMatrix4x4()
        inputVolume.GetRASToIJKMatrix(rasToIjk)

        # Convert the scanline points to IJK coordinates
        scanlinePoints_IJK = np.zeros((2, 4))
        for i in range(2):
            rasToIjk.MultiplyPoint(scanlinePoints_RAS[i], scanlinePoints_IJK[i])

        # Check all sequences of all sequence browsers in the scene to find which one uses inputVolume as proxy node
        sequenceBrowsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        sequenceNode = None
        sequenceBrowser = None
        for sequenceBrowser in sequenceBrowsers:
            collection = vtk.vtkCollection()
            sequenceBrowser.GetSynchronizedSequenceNodes(collection, True)
            for i in range(collection.GetNumberOfItems()):
                sequenceNode = collection.GetItemAsObject(i)
                proxyNode = sequenceBrowser.GetProxyNode(sequenceNode)
                if proxyNode.GetID() == inputVolume.GetID():
                    break
                
        # Prepare np array to store the time series
        numFrames = sequenceBrowser.GetNumberOfItems()
        singleFrameArray = slicer.util.arrayFromVolume(inputVolume)
        if len(singleFrameArray.shape) == 4:
            singleFrameArray = singleFrameArray[0, :, :, :]  # Eliminate the first dimension
        
        # Prepare np array to store the time series
        scanlineLength_px = int(round(np.linalg.norm(scanlinePoints_IJK[1, :3] - scanlinePoints_IJK[0, :3])))
        mmodeArray = np.zeros((1, scanlineLength_px, numFrames), dtype=singleFrameArray.dtype)  # np.array should be in rows, columns order

        for itemIndex in range(sequenceBrowser.GetNumberOfItems()):
            sequenceBrowser.SetSelectedItemNumber(itemIndex)
            singleFrameArray = slicer.util.arrayFromVolume(inputVolume)
            if len(singleFrameArray.shape) == 4:
                singleFrameArray = singleFrameArray[0, :, :, :]
            mmodeArray[0, :, itemIndex] = self.sample_line(singleFrameArray, scanlinePoints_IJK[0, :3], scanlinePoints_IJK[1, :3], num_points=scanlineLength_px, average_channels=True)
            slicer.app.processEvents()
        
        # Compute horizontal scaling factor
        horizontalPixelSpacing_mm = paperSpeed_mm_per_px/20.0  # Assuming 20 FPS cine rate
        horizontalPixelsPerFrame = horizontalPixelSpacing_mm / inputVolume.GetSpacing()[0]
        
        # Compute alternative horizontal scaling factor for square output
        horizontalPixelsPerFrame_square = scanlineLength_px / numFrames
        
        # Keep the larger of the two scaling factors
        horizontalPixelsPerFrame = max(horizontalPixelsPerFrame, horizontalPixelsPerFrame_square)
        
        # Scale mmodeArray along the time axis using linear interpolation
        zoom_factors = [1, 1, horizontalPixelsPerFrame]
        mmodeArray_scaled = zoom(mmodeArray, zoom_factors, order=1)
        
        # Flip the mmodeArray_scaled along the time axis
        mmodeArray_scaled = np.flip(mmodeArray_scaled, axis=2)
        
        # Create output imagedata with the right dimensions and the same pixel type as the  input volume
        outputImageData = vtk.vtkImageData()
        outputImageData.SetDimensions(mmodeArray_scaled.shape[2], scanlineLength_px, 1)
        outputImageData.AllocateScalars(inputVolume.GetImageData().GetScalarType(), 1)

        # Set outputImageData as the image data of the outputVolume
        outputVolume.SetAndObserveImageData(outputImageData)
        
        # Update the output volume with the mmodeArray
        slicer.util.updateVolumeFromArray(outputVolume, mmodeArray_scaled)

        # Make sure that the output volume has the same spacing as the input volume along the scanline, and increase the spacing by a factor along time
        outputVolume.SetSpacing([inputVolume.GetSpacing()[0], inputVolume.GetSpacing()[1], inputVolume.GetSpacing()[2]])

        # Set up default window/level for display
        outputDisplayNode = outputVolume.GetDisplayNode()
        if outputDisplayNode is None:
            outputVolume.CreateDefaultDisplayNodes()
            outputDisplayNode = outputVolume.GetDisplayNode()
        outputDisplayNode.SetWindow(255)
        outputDisplayNode.SetLevel(127)
        
        stopTime = time.time()
        logging.info(f"M-mode conversion completed in {stopTime-startTime:.2f} seconds")
    
    def getCurrentSequenceBrowser(self):
        """
        Get the sequence browser that contains a sequence with the currently selected input volume as a proxy node.
        :return: Sequence browser node
        """
        inputVolume = self.getParameterNode().inputVolume
        if not inputVolume:
            return None
        
        sequenceBrowsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        for sequenceBrowser in sequenceBrowsers:
            collection = vtk.vtkCollection()
            sequenceBrowser.GetSynchronizedSequenceNodes(collection, True)
            for i in range(collection.GetNumberOfItems()):
                sequenceNode = collection.GetItemAsObject(i)
                proxyNode = sequenceBrowser.GetProxyNode(sequenceNode)
                if proxyNode.GetID() == inputVolume.GetID():
                    return sequenceBrowser
        return None
    
    def saveMeasurement(self, outputDirectory, imageFilename=None):
        """
        Load measurements dataframe from a "measurements.csv" file in the output directory.
        If it doesn't exist, create a new dataframe.
        Add the current measurement to the dataframe and save it back to the file.
        
        :param outputDirectory: Directory where the measurements file is saved.
        :returns: DataFrame with the measurements.
        """
        measurementsFile = os.path.join(outputDirectory, "measurements.csv")
        
        if os.path.exists(measurementsFile):
            measurements = pd.read_csv(measurementsFile)
        else:
            measurements = pd.DataFrame(columns=["PatientName", "PatientID", "DicomInstanceUID", "Timestamp", "Length", "ImageFilename"])
        
        # Check all sequences of all sequence browsers in the scene to find which one uses inputVolume as proxy node
        sequenceBrowser = self.getCurrentSequenceBrowser()
        if not sequenceBrowser:
            logging.error("No sequence browser found with the current input volume as a proxy node")
            return measurements
        
        dicomData = self.getDicomDataForBrowser(sequenceBrowser)
        if dicomData is None:
            logging.error("No DICOM data found for the current sequence browser")
            return measurements
        
        measurementLine = self.getParameterNode().measurementMarkup
        measurementLineCoordinates = np.zeros((2, 3))
        if measurementLine and measurementLine.GetNumberOfControlPoints() > 1:
            measurementLine.GetNthControlPointPositionWorld(0, measurementLineCoordinates[0])
            measurementLine.GetNthControlPointPositionWorld(1, measurementLineCoordinates[1])
            # Compute the length of the measurement line
            length = np.linalg.norm(measurementLineCoordinates[1] - measurementLineCoordinates[0])
            length = f"{length:.3f}"  # Convert length to a string with three decimal places
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            # Get the patient name, ID, and DICOM instance UID
            patientName = dicomData.PatientName
            patientID = dicomData.PatientID
            dicomInstanceUID = dicomData.SOPInstanceUID
            imageBasename = os.path.basename(imageFilename)
            new_row = pd.DataFrame({"PatientName": [patientName], "PatientID": [patientID], "DicomInstanceUID": [dicomInstanceUID], "Timestamp": [timestamp],
                                    "Length": [length], "ImageFilename": [imageBasename]})
            if measurements.empty:
                measurements = new_row
            else:
                measurements = pd.concat([measurements, new_row], ignore_index=True)
            # Save the dataframe to the file
            try:
                measurements.to_csv(measurementsFile, index=False)
                logging.info(f"Measurement saved to {measurementsFile}")
            except Exception as e:
                logging.error(f"Failed to save measurements: {e}")
                # Pop up a dialog to inform the user
                qt.QMessageBox.warning(slicer.util.mainWindow(), "Save measurements", f"Failed to save measurements: {e}")
        
        return measurements
    
    def saveMarkedImage(self, volumeNode, markupsLineNode, filepath, flip=False):
        """
        Save an image with a line drawn on it.
        :param volumeNode: Volume node
        :param markupsLineNode: Markups line node
        :param filepath: Path to save the image. E.g. "/path/to/image.png"
        """
        input_array = slicer.util.arrayFromVolume(volumeNode)
        
        # Drop the frames dimension from the input array
        if len(input_array.shape) == 4:  # Color image
            input_array = input_array[0, :, :, :]
        else:  # Grayscale image
            input_array = input_array[0, :, :]
        
        # Add the line as a white line on the image
        scanlineCoordinates = np.zeros((2, 3))
        if markupsLineNode and markupsLineNode.GetNumberOfControlPoints() > 1:
            markupsLineNode.GetNthControlPointPositionWorld(0, scanlineCoordinates[0])
            markupsLineNode.GetNthControlPointPositionWorld(1, scanlineCoordinates[1])
            # Convert the scanline coordinates to IJK coordinates
            rasToIjk = vtk.vtkMatrix4x4()
            volumeNode.GetRASToIJKMatrix(rasToIjk)
            scanlineCoordinates_IJK = np.zeros((2, 4))
            for i in range(2):
                homogenousRAS = np.append(scanlineCoordinates[i], 1)
                rasToIjk.MultiplyPoint(homogenousRAS, scanlineCoordinates_IJK[i])
            
            # Draw the scanline on the input image
            if len(input_array.shape) == 4:  # Color image
                for i in range(3):
                    input_array[:, :, i] = self.draw_line(input_array[:, :, i], scanlineCoordinates_IJK[0], scanlineCoordinates_IJK[1], value=255)
            else:  # Grayscale image
                input_array = self.draw_line(input_array, scanlineCoordinates_IJK[0], scanlineCoordinates_IJK[1], value=255)
        
        # Flip outputArray along the vertical and horizontal axes
        if flip:
            input_array = np.flip(input_array, axis=0)
            input_array = np.flip(input_array, axis=1)
        
        # Save the image
        cv2.imwrite(filepath, input_array)
    
    def saveAnnotatedMmodeImage(self, filepath: str) -> None:
        """
        Save the output volume as an annotated M-mode image.
        :param outputVolume: M-mode volume
        :param filepath: Path to save the image
        """
        # Add "_bmode" to the filename before the extension for saving the original image
        filename_stem = os.path.splitext(filepath)[0]
        bmode_filename = filename_stem + "_bmode.png"
        input_volume = self.getParameterNode().inputVolume
        scanlineMarkup = self.getParameterNode().scanlineMarkup
        self.saveMarkedImage(input_volume, scanlineMarkup, bmode_filename)
        logging.info(f"B-mode frame saved to {bmode_filename}")
            
        # Get the output volume as a numpy array
        outputVolume = self.getParameterNode().mmodeVolume
        measurementLine = self.getParameterNode().measurementMarkup
        self.saveMarkedImage(outputVolume, measurementLine, filepath, flip=True)
        logging.info(f"Annotated M-mode image saved to {filepath}")
            
    def draw_line(self, image_array, p1, p2, value=255):
        """
        Draw a line on an image array.
        :param image_array: 2D numpy array
        :param p1: Start point of the line
        :param p2: End point of the line
        :param value: Value to set along the line
        :return: Image array with the line drawn
        """
        # Use cv2 to draw the line
        cv_array = cv2.UMat(image_array)
        annotated_cv_image = cv2.line(cv_array, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), value, 2)
        annotated_image = annotated_cv_image.get()
        return annotated_image
        

#
# MmodeAnalysisTest
#


class MmodeAnalysisTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MmodeAnalysis1()

    def test_MmodeAnalysis1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        self.delayDisplay("Test passed")
