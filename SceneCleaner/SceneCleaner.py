import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSequenceBrowserNode, vtkMRMLModelNode, vtkMRMLLinearTransformNode, vtkMRMLMarkupsFiducialNode


#
# SceneCleaner
#


class SceneCleaner(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Scene Cleaner")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
                                 <p>
                                 This module cleans the scene by removing unnecessary nodes, setting up transformation hierarchy, and renaming nodes.<br>
                                 The input sequence browser node and the ultarsound image node must be selected. Other selections are optional.
                                 The module will remove all noded that are not selected.
                                 </p>
                                 <p>
                                 See more information in <a href="https://github.com/SlicerUltrasound/SlicerUltrasound?tab=readme-ov-file#ultrasound-extension">extension documentation</a>.
                                 </p>
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

    # SceneCleaner1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SceneCleaner",
        sampleName="SceneCleaner1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "SceneCleaner1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SceneCleaner1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SceneCleaner1",
    )

    # SceneCleaner2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SceneCleaner",
        sampleName="SceneCleaner2",
        thumbnailFileName=os.path.join(iconsPath, "SceneCleaner2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SceneCleaner2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SceneCleaner2",
    )


#
# SceneCleanerParameterNode
#


@parameterNodeWrapper
class SceneCleanerParameterNode:
    """
    The parameters needed by module.

    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputSequenceBrowser: vtkMRMLSequenceBrowserNode
    # imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    ctVolume: vtkMRMLScalarVolumeNode
    usVolume: vtkMRMLScalarVolumeNode
    landmarksMarkup: vtkMRMLMarkupsFiducialNode
    atlasModel: vtkMRMLModelNode
    usImage: vtkMRMLScalarVolumeNode
    ctToUsTransform: vtkMRMLLinearTransformNode


#
# SceneCleanerWidget
#


class SceneCleanerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SceneCleaner.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SceneCleanerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.cleanNodesButton.connect("clicked(bool)", self.onCleanNodesButton)

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

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        
    def setParameterNode(self, inputParameterNode: Optional[SceneCleanerParameterNode]) -> None:
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

    def _onParameterNodeModified(self, caller=None, event=None) -> None:
        if self._parameterNode is None:
            return
        
        if self._parameterNode.inputSequenceBrowser and self._parameterNode.usImage:
            self.ui.cleanNodesButton.toolTip = _("Clean nodes")
            self.ui.cleanNodesButton.enabled = True
        else:
            self.ui.cleanNodesButton.toolTip = _("Select input nodes")
            self.ui.cleanNodesButton.enabled = False

    def onCleanNodesButton(self) -> None:
        """Callback function for clean nodes button."""
        scanName = self.ui.scanNameComboBox.currentText
        patientId = self.ui.patientIdLineEdit.text
        self.logic.cleanNodes(scanName, patientId)


#
# SceneCleanerLogic
#


class SceneCleanerLogic(ScriptedLoadableModuleLogic):
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
        return SceneCleanerParameterNode(super().getParameterNode())
    
    def _RemoveUnusedSequenceBrowsers(self) -> None:
        """
        Remove sequence browsers with all their sequences that are not used in the scene.
        :param sequenceBrowserNode: sequence browser node to keep with all its sequences
        """
        parameterNode = self.getParameterNode()
        sequenceBrowserNode = parameterNode.inputSequenceBrowser
        
        if not sequenceBrowserNode:
            raise ValueError("Sequence browser node is invalid")
        
        # Remove sequences that are not used by the given sequence browser
        
        sequencesToKeep = vtk.vtkCollection()
        sequenceBrowserNode.GetSynchronizedSequenceNodes(sequencesToKeep, True)
        
        allSequences = slicer.util.getNodesByClass("vtkMRMLSequenceNode")
        numSequencesRemoved = 0
        for sequenceNode in allSequences:
            if not sequencesToKeep.IsItemPresent(sequenceNode):
                sequenceNode.RemoveAllDataNodes()
                slicer.mrmlScene.RemoveNode(sequenceNode)
                numSequencesRemoved += 1
        
        logging.info(f"Removed {numSequencesRemoved} unused sequence nodes")
        
        # Remove sequence browsers that is not given as an argument
        
        allSequenceBrowserNodes = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        numSequenceBrowsersRemoved = 0
        for sequenceBrowser in allSequenceBrowserNodes:
            if sequenceBrowser != sequenceBrowserNode:
                slicer.mrmlScene.RemoveNode(sequenceBrowser)
                numSequenceBrowsersRemoved += 1
        
        logging.info(f"Removed {numSequenceBrowsersRemoved} unused sequence browser nodes")
    
    def _RemoveUnusedModels(self) -> None:
        """
        Remove models that are not used in the scene.
        """
        atlasModel = self.getParameterNode().atlasModel
        allModels = slicer.util.getNodesByClass("vtkMRMLModelNode")
        numModelsRemoved = 0
        for modelNode in allModels:
            if atlasModel is not None and modelNode.GetID() != atlasModel.GetID():
                slicer.mrmlScene.RemoveNode(modelNode)
                numModelsRemoved += 1
        
        logging.info(f"Removed {numModelsRemoved} unused models")    
    
    def _RemoveUnusedMarkups(self) -> None:
        """
        Remove markups that are not used in the scene.
        """
        landmarksMarkup = self.getParameterNode().landmarksMarkup
        allMarkups = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        numMarkupsRemoved = 0
        for markupNode in allMarkups:
            if landmarksMarkup is not None and markupNode.GetID() != landmarksMarkup.GetID():
                slicer.mrmlScene.RemoveNode(markupNode)
                numMarkupsRemoved += 1
        
        # Remove all curve markups
        allMarkups = slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode")
        for markupNode in allMarkups:
            slicer.mrmlScene.RemoveNode(markupNode)
            numMarkupsRemoved += 1
        
        # Remove all ROI markup nodes, unless their name contains "Reconstructor"
        allMarkups = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for markupNode in allMarkups:
            if "Reconstructor" not in markupNode.GetName():
                slicer.mrmlScene.RemoveNode(markupNode)
                numMarkupsRemoved += 1
        
        logging.info(f"Removed {numMarkupsRemoved} unused markups")
    
    def cleanNodes(self, scanName, patientId=None) -> None:
        """
        Based on parameter node references, clean the scene from unnecessary nodes and set up transformation hierarchy.
        :param scanName: string to code scan (axial or sagittal)
        :param patientId: string to code patient
        """
        parameterNode = self.getParameterNode()
        
        if not parameterNode.usImage:
            raise ValueError("Input US image is invalid")
        
        self._RemoveUnusedSequenceBrowsers()
        
        # Rename atlas model if it is set
        if parameterNode.atlasModel:
            parameterNode.atlasModel.SetName(f"{patientId}_{scanName}_AtlasModel")
        
        self._RemoveUnusedModels()
        
        # Rename markups if set
        if parameterNode.landmarksMarkup:
            parameterNode.landmarksMarkup.SetName(f"{patientId}_{scanName}_Landmarks")
        
        self._RemoveUnusedMarkups()


#
# SceneCleanerTest
#


class SceneCleanerTest(ScriptedLoadableModuleTest):
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
        self.test_SceneCleaner1()

    def test_SceneCleaner1(self):
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

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("SceneCleaner1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SceneCleanerLogic()

        # Test algorithm with non-inverted threshold
        logic.cleanNodes(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.cleanNodes(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
