import logging
import os
from typing import Annotated, Optional

import vtk, qt, slicer
import logging
import numpy as np


import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import SampleData

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLSequenceBrowserNode
from slicer import vtkMRMLSegmentationNode
from slicer import vtkMRMLModelNode


#
# TimeSeriesAnnotation
#


class TimeSeriesAnnotation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Time Series Annotation")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = ["VolumeResliceDriver"]  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This module is to annotate time series ultrasound images.<br>
See more information in <a href="https://github.com/SlicerUltrasound/SlicerUltrasound">module documentation</a>.
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

    # TimeSeriesAnnotation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="TimeSeriesAnnotation",
        sampleName="TimeSeriesAnnotation1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "TimeSeriesAnnotation1.png"),
        # Download URL and target file name
        uris="https://onedrive.live.com/download?resid=7230D4DEC6058018%21114824&authkey=!AKoTnKwt3AP5OiU",
        fileNames="1133_LandmarkingScan_Sa_Test.mrb",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:4348c089b9ca9f25c0a70dff1ace1b94cab66b2b08c7524ddf638d4a32a4e57d",
        # This node name will be used when the data set is loaded
        nodeNames="TimeSeriesAnnotation1",
    )

#
# TimeSeriesAnnotationParameterNode
#

@parameterNodeWrapper
class TimeSeriesAnnotationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputBrowser: vtkMRMLSequenceBrowserNode
    inputVolume: vtkMRMLScalarVolumeNode
    inputSkipNumber: int = 4
    segmentationBrowser: vtkMRMLSequenceBrowserNode
    segmentation: vtkMRMLSegmentationNode
    ultrasoundModel: vtkMRMLModelNode
    showUltrasoundModel: bool = True
    showOverlay: bool = True
    reviseSegmentations: bool = False
    
#
# TimeSeriesAnnotationWidget
#

class TimeSeriesAnnotationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        self._updatingGuiFromParameterNode = False
        
        def setup(self):
            # Register subject hierarchy plugin
            import SubjectHierarchyPlugins
            scriptedPlugin = slicer.qSlicerSubjectHierarchyScriptedPlugin(None)
            scriptedPlugin.setPythonSource(SubjectHierarchyPlugins.SegmentEditorSubjectHierarchyPlugin.filePath)

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/TimeSeriesAnnotation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)
        
        # Configure segment editor widget
        self.ui.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentEditorWidget.setEffectNameOrder(['Threshold', 'Paint', 'Erase', 'Margin',
                                                        'Islands', 'Smoothing', 'Merge', 'Logical operators'])
        self.ui.segmentEditorWidget.unorderedEffectsVisible = False
        
        # Not sure if this is needed
        import qSlicerSegmentationsEditorEffectsPythonQt
        factory = qSlicerSegmentationsEditorEffectsPythonQt.qSlicerSegmentEditorEffectFactory()
        self.effectFactorySingleton = factory.instance()
        self.effectFactorySingleton.connect('effectRegistered(QString)', self.editorEffectRegistered)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = TimeSeriesAnnotationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

        # Buttons
        self.ui.captureButton.connect("clicked(bool)", self.onCaptureButton)
        self.ui.sampleDataButton.connect("clicked(bool)", self.onSampleDataButton)
        self.ui.sliceViewButton.connect("toggled(bool)", self.onSliceViewButton)
        self.ui.reviseButton.connect("toggled(bool)", self.onReviseButton)
        self.ui.overlayButton.connect("toggled(bool)", self.onOverlayButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
        
        # Collapse DataProbe widget
        mw = slicer.util.mainWindow()
        if mw:
            w = slicer.util.findChild(mw, "DataProbeCollapsibleWidget")
            if w:
                w.collapsed = True
                
        # Make sure sequences toolbar is visible
        slicer.modules.sequences.toolBar().setVisible(True)

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()
    
    def onSceneEndImport(self, caller, event) -> None:
        """Called after a scene is imported."""
        # If this module is shown while the scene is imported then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()
        
    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        # Make sure segment editor node is created and set before modifying it from parameters
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if not segmentEditorNode:
            segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", segmentEditorSingletonTag)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
        self.ui.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)            

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[TimeSeriesAnnotationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
            self._onParameterNodeModified()
        
    def editorEffectRegistered(self):
        self.editor.updateEffectList()
    
    def _onParameterNodeModified(self, caller=None, event=None) -> None:
        """
        Update GUI when parameter node is changed.
        :param caller: the parameter node (unused)
        :param event: modified event that triggered the update (unused)
        :return: None
        """
        
        # Prevent infinite loop if GUI is modifying parameter node
        if self._updatingGuiFromParameterNode:
            return
        self._updatingGuiFromParameterNode = True
        
        if self._parameterNode and self._parameterNode.inputBrowser:
            if self._parameterNode.reviseSegmentations == False:
                slicer.modules.sequences.toolBar().setActiveBrowserNode(self._parameterNode.inputBrowser)
            elif self._parameterNode.segmentationBrowser:
                slicer.modules.sequences.toolBar().setActiveBrowserNode(self._parameterNode.segmentationBrowser)
        
        # Update buttons states and tooltips
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.inputBrowser and self._parameterNode.segmentation:
            self.ui.captureButton.toolTip = _("Record current frame")
            self.ui.captureButton.enabled = True
            self.ui.skipButton.toolTip = _("Skip current frame without recording")
            self.ui.skipButton.enabled = True
            self.ui.deleteButton.toolTip = _("Delete segmentation on current frame")
            self.ui.deleteButton.enabled = True
        else:
            self.ui.captureButton.toolTip = _("Select inputs to enable this button")
            self.ui.captureButton.enabled = False
            self.ui.skipButton.toolTip = _("Select inputs to enable this button")
            self.ui.skipButton.enabled = False
            self.ui.deleteButton.toolTip = _("Select inputs to enable this button")
            self.ui.deleteButton.enabled = False
        
        if self._parameterNode.reviseSegmentations:
            self.ui.reviseButton.checked = True
            self.ui.reviseButton.text = "Stop revising"
        else:
            self.ui.reviseButton.checked = False
            self.ui.reviseButton.text = "Revise segmentations"
        
        # Update selected segmentation node in segment editor
        if self._parameterNode and self._parameterNode.segmentation:
            if self.ui.segmentEditorWidget.segmentationNode() != self._parameterNode.segmentation:
                print(f"Setting segmentation node: {self._parameterNode.segmentation.GetID()}")
                self.ui.segmentEditorWidget.setSegmentationNode(self._parameterNode.segmentation)
        
        # Use selected input volume in segment editor and in 2D views
        if self._parameterNode and self._parameterNode.inputVolume:
            layoutManager = slicer.app.layoutManager()
            for sliceViewName in layoutManager.sliceViewNames():
                sliceWidget = layoutManager.sliceWidget(sliceViewName)
                sliceWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self._parameterNode.inputVolume.GetID())
            self.ui.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.inputVolume)
        
        # Show overlay foreground volume in 2D views
        layoutManager = slicer.app.layoutManager()
        compositeNode = layoutManager.sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        if self._parameterNode and self._parameterNode.showOverlay:
            compositeNode.SetForegroundOpacity(0.3)
            self.ui.overlayButton.checked = True
        else:
            compositeNode.SetForegroundOpacity(0.0)
            self.ui.overlayButton.checked = False
        
        # Update ultrasound slice representation in 3D view
        layoutManager = slicer.app.layoutManager()
        redWidget = layoutManager.sliceWidget('Red')
        redWidget.sliceController().setSliceVisible(self._parameterNode.showUltrasoundModel)
        if self._parameterNode and self._parameterNode.ultrasoundModel:
            self._parameterNode.ultrasoundModel.GetDisplayNode().SetVisibility(not self._parameterNode.showUltrasoundModel)
        
        self._updatingGuiFromParameterNode = False

    def onCaptureButton(self) -> None:
        logging.info("onCaptureButton")
        if not self._parameterNode:
            logging.error("Parameter node is invalid")
            return
        
        self.logic.captureCurrentFrame()
        

    def onSampleDataButton(self) -> None:
        logging.info("onSampleDataButton")
        """Load sample data when user clicks "Load Sample Data" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to load sample data.")):
            SampleData.SampleDataLogic().downloadSample("TimeSeriesAnnotation1")
    
    def onSliceViewButton(self, checked) -> None:
        """Show/hide slice views when user clicks "Show Slice Views" button."""
        self._parameterNode.showUltrasoundModel = checked
    
    def onReviseButton(self, checked) -> None:
        """Show/hide slice views when user clicks "Show Slice Views" button."""
        self._parameterNode.reviseSegmentations = checked
    
    def onOverlayButton(self, checked) -> None:
        """Show/hide slice views when user clicks "Show Slice Views" button."""
        self._parameterNode.showOverlay = checked
            
#
# TimeSeriesAnnotationLogic
#


class TimeSeriesAnnotationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    ATTRIBUTE_PREFIX = 'SingleSliceSegmentation_'
    ORIGINAL_IMAGE_INDEX = ATTRIBUTE_PREFIX + 'OriginalImageIndex'
    
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return TimeSeriesAnnotationParameterNode(super().getParameterNode())
    
    def captureCurrentFrame(self):
        """
        Record the current frame and segmentation in the segmentation browser.
        """
        parameterNode = self.getParameterNode()
        originalIndexStr = parameterNode.segmentation.GetAttribute(self.ORIGINAL_IMAGE_INDEX)
        if not parameterNode.reviseSegmentations:  # Adding new segmentation to the record
            inputIndex = parameterNode.inputBrowser.GetSelectedItemNumber()
            parameterNode.segmentation.SetAttribute(self.ORIGINAL_IMAGE_INDEX, str(inputIndex))
            
        
        #TODO: Implement this
        
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
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# TimeSeriesAnnotationTest
#


class TimeSeriesAnnotationTest(ScriptedLoadableModuleTest):
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
        self.test_TimeSeriesAnnotation1()

    def test_TimeSeriesAnnotation1(self):
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

        
        registerSampleData()
        inputVolume = SampleData.downloadSample("TimeSeriesAnnotation1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = TimeSeriesAnnotationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
