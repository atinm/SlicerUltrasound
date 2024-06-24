import logging
import numpy as np
import os
from typing import Annotated, Optional

import qt
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

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVolumeReconstructionNode, vtkMRMLMarkupsFiducialNode, vtkMRMLModelNode, vtkMRMLLinearTransformNode


#
# NeedleGuide
#


class NeedleGuide(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("NeedleGuide")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Ultrasound")]
        self.parent.dependencies = ["VolumeResliceDriver"]  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#NeedleGuide">module documentation</a>.
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

    # NeedleGuide1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="NeedleGuide",
        sampleName="NeedleGuide1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "NeedleGuide1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="NeedleGuide1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="NeedleGuide1",
    )

    # NeedleGuide2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="NeedleGuide",
        sampleName="NeedleGuide2",
        thumbnailFileName=os.path.join(iconsPath, "NeedleGuide2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="NeedleGuide2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="NeedleGuide2",
    )


#
# NeedleGuideParameterNode
#


@parameterNodeWrapper
class NeedleGuideParameterNode:
    """
    The parameters needed by module.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    referenceToRas: vtkMRMLLinearTransformNode
    stylusToReference: vtkMRMLLinearTransformNode
    predictionVolume: vtkMRMLScalarVolumeNode
    reconstructorNode: vtkMRMLVolumeReconstructionNode
    targetMarkups: vtkMRMLMarkupsFiducialNode
    needleGuideMarkups: vtkMRMLMarkupsFiducialNode
    blurSigma: Annotated[float, WithinRange(0, 5)] = 0.5
    reconstructedVolume: vtkMRMLScalarVolumeNode
    opacityThreshold: Annotated[int, WithinRange(-100, 200)] = 60
    invertThreshold: bool = False
    targetModel: vtkMRMLModelNode
    targetRadius: Annotated[float, WithinRange(0, 10)] = 2.5
    targetCoordinatesRas: tuple[float, float, float]

#
# NeedleGuideWidget
#

class NeedleGuideWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    
    LAYOUT_2D3D = 601

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        self.observedTargetMarkups = None
        self.observedNeedleGuideMarkups = None

        # for debugging
        slicer.mymod = self

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/NeedleGuide.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = NeedleGuideLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # UI widget connections
        self.ui.applyButton.connect("clicked(bool)", self.onReconstructionButton)
        self.ui.volumeOpacitySlider.connect("valueChanged(int)", self.onVolumeOpacitySlider)
        self.ui.setRoiButton.connect("clicked(bool)", self.onSetRoiButton)
        self.ui.targetRadiusSlider.connect("valueChanged(double)", self.onTargetRadiusSlider)
        
        # Add custom layout
        self.addCustomLayouts()
        slicer.app.layoutManager().setLayout(self.LAYOUT_2D3D)
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
        # Collapse DataProbe widget
        mw = slicer.util.mainWindow()
        if mw:
            w = slicer.util.findChild(mw, "DataProbeCollapsibleWidget")
            if w:
                w.collapsed = True
    
    def addCustomLayouts(self):
        layout2D3D = \
        """
        <layout type="horizontal" split="true">
            <item splitSize="500">
            <view class="vtkMRMLViewNode" singletontag="1">
                <property name="viewlabel" action="default">1</property>
            </view>
            </item>
            <item splitSize="500">
            <view class="vtkMRMLSliceNode" singletontag="Red">
                <property name="orientation" action="default">Axial</property>
                <property name="viewlabel" action="default">R</property>
                <property name="viewcolor" action="default">#F34A33</property>
            </view>
            </item>
        </layout>
        """
         
        layoutManager = slicer.app.layoutManager()
        if not layoutManager.layoutLogic().GetLayoutNode().SetLayoutDescription(self.LAYOUT_2D3D, layout2D3D):
            layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(self.LAYOUT_2D3D, layout2D3D)
        
        # Add button to layout selector toolbar for this custom layout
        viewToolBar = slicer.util.mainWindow().findChild("QToolBar", "ViewToolBar")
        layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
        layoutSwitchActionParent = layoutMenu  # use `layoutMenu` to add inside layout list, use `viewToolBar` to add next the standard layout list
        layoutSwitchAction = layoutSwitchActionParent.addAction("3D-2D") # add inside layout list
        layoutSwitchAction.setData(self.LAYOUT_2D3D)
        layoutSwitchAction.setIcon(qt.QIcon(":Icons/Go.png"))
        layoutSwitchAction.setToolTip("3D and slice view")
    
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
        
        targetModel = self._parameterNode.targetModel
        if targetModel is None:
            targetModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TargetModel")
            targetModel.CreateDefaultDisplayNodes()
            displayNode = targetModel.GetDisplayNode()
            displayNode.SetColor(0.0, 1.0, 0.0)
            displayNode.BackfaceCullingOff()
            displayNode.Visibility2DOn()
            displayNode.Visibility3DOn()
            displayNode.SetSliceIntersectionThickness(3)
            displayNode.SetOpacity(0.6)
            self._parameterNode.targetModel = targetModel

        
    def setParameterNode(self, inputParameterNode: Optional[NeedleGuideParameterNode]) -> None:
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
        """
        Update GUI based on parameter node changes.
        """
        # Update volume reconstruction button
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.predictionVolume and self._parameterNode.reconstructorNode:
            if self.logic.reconstructing:
                self.ui.applyButton.text = _("Stop volume reconstruction")
                self.ui.applyButton.toolTip = _("Stop volume reconstruction")
                self.logic.startVolumeReconstruction()
            else:
                self.ui.applyButton.text = _("Start volume reconstruction")
                self.ui.applyButton.toolTip = _("Start volume reconstruction")
                self.logic.stopVolumeReconstruction()
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input nodes to enable volume reconstruction")
            self.ui.applyButton.enabled = False
        
        # Update opacity threshold slider
        if self._parameterNode and self._parameterNode.reconstructedVolume:
            self.ui.volumeOpacitySlider.enabled = True
        else:
            self.ui.volumeOpacitySlider.enabled = False
            
        # Set and observe target markups node
        if self.observedTargetMarkups != self._parameterNode.targetMarkups:
            if self.observedTargetMarkups:
                self.removeObserver(self.observedTargetMarkups, vtk.vtkCommand.ModifiedEvent, self._onTargetMarkupsModified)
            self.observedTargetMarkups = self._parameterNode.targetMarkups
            if self.observedTargetMarkups:
                self.addObserver(self.observedTargetMarkups, vtk.vtkCommand.ModifiedEvent, self._onTargetMarkupsModified)
            self._onTargetMarkupsModified()

        # Set and observe StylusToReference transform
        if self.observedNeedleGuideMarkups != self._parameterNode.needleGuideMarkups:
            if self.observedNeedleGuideMarkups:
                self.removeObserver(self.observedNeedleGuideMarkups, vtkMRMLMarkupsFiducialNode.TransformModifiedEvent, self._onStylusToReferenceModified)
            self.observedNeedleGuideMarkups = self._parameterNode.needleGuideMarkups
            if self.observedNeedleGuideMarkups:
                self.addObserver(self.observedNeedleGuideMarkups, vtkMRMLMarkupsFiducialNode.TransformModifiedEvent, self._onStylusToReferenceModified)
            self._onStylusToReferenceModified()
        
        # Set up targets table
        self.ui.targetTableWidget.setColumnCount(1)
        self.ui.targetTableWidget.setHorizontalHeaderLabels(["Target"])
        header = self.ui.targetTableWidget.horizontalHeader()
        header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        self.ui.targetTableWidget.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.ui.targetTableWidget.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.ui.targetTableWidget.itemSelectionChanged.connect(self.onTargetSelectionChanged)
        
    def _onTargetMarkupsModified(self, caller=None, event=None) -> None:
        """
        Update GUI based on target markups changes.
        """
        # Update target table from target markup names
        self.ui.targetTableWidget.setRowCount(0)
        if self._parameterNode and self._parameterNode.targetMarkups:
            for i in range(self._parameterNode.targetMarkups.GetNumberOfControlPoints()):
                self.ui.targetTableWidget.insertRow(i)
                self.ui.targetTableWidget.setItem(i, 0, qt.QTableWidgetItem(self._parameterNode.targetMarkups.GetNthControlPointLabel(i)))
    
    def onTargetSelectionChanged(self) -> None:
        """
        Update GUI based on target selection changes.
        """
        logging.info("Target selection changed")
        selectedRow = self.ui.targetTableWidget.currentRow()
        
        # Clear targetModel mesh
        targetModel = self._parameterNode.targetModel
        if targetModel:
            targetModel.SetAndObservePolyData(None)
        
        if selectedRow < 0:
            return
        
        logging.info(f"Selected row: {selectedRow}")
        
        selectedPointPositionRas = np.zeros(3)
        for i in range(self._parameterNode.targetMarkups.GetNumberOfControlPoints()):
            if i == selectedRow:
                self._parameterNode.targetMarkups.SetNthControlPointSelected(i, False)
                self._parameterNode.targetMarkups.GetNthControlPointPosition(i, selectedPointPositionRas)
            else:
                self._parameterNode.targetMarkups.SetNthControlPointSelected(i, True)
        
        # Create a sphere at the selected point
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(self._parameterNode.targetRadius)
        sphereSource.SetCenter(selectedPointPositionRas)
        sphereSource.SetPhiResolution(12)
        sphereSource.SetThetaResolution(24)
        innerSphereSource = vtk.vtkSphereSource()
        innerSphereSource.SetRadius(self._parameterNode.targetRadius*0.5)
        innerSphereSource.SetCenter(selectedPointPositionRas)
        innerSphereSource.SetPhiResolution(10)
        innerSphereSource.SetThetaResolution(20)
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputConnection(sphereSource.GetOutputPort())
        appendFilter.AddInputConnection(innerSphereSource.GetOutputPort())
        appendFilter.Update()
        
        targetModel.SetAndObservePolyData(appendFilter.GetOutput())

        # save selected fiducial for distance measurement
        self._parameterNode.targetCoordinatesRas = tuple(selectedPointPositionRas)
        self._onStylusToReferenceModified()

    def _onStylusToReferenceModified(self, caller=None, event=None) -> None:
        if (self._parameterNode.needleGuideMarkups.GetNumberOfControlPoints() > 0
            and self._parameterNode.targetCoordinatesRas is not None
        ):
            # Compute distance between needle guide and target
            needleGuideCoordsRas = np.zeros(3)
            self._parameterNode.needleGuideMarkups.GetNthControlPointPositionWorld(0, needleGuideCoordsRas)
            distance = np.linalg.norm(needleGuideCoordsRas - self._parameterNode.targetCoordinatesRas)

            # Update corner annotation
            layoutManager = slicer.app.layoutManager()
            for i in range(layoutManager.threeDViewCount):
                view = layoutManager.threeDWidget(i).threeDView()
                view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft, f"Depth: {distance:.0f}mm")
                view.forceRender()
    
    def onTargetRadiusSlider(self, value: float) -> None:
        """
        Update target radius.
        """
        self.onTargetSelectionChanged()
    
    def onReconstructionButton(self) -> None:
        """Run processing when user clicks button."""
        # Start volume reconstruction if not already started. Stop otherwise.
        
        if self.logic.reconstructing:
            self.logic.stopVolumeReconstruction()
        else:
            self.logic.startVolumeReconstruction()
    
    def onVolumeOpacitySlider(self, value: int) -> None:
        """Update volume rendering opacity threshold."""
        if self._parameterNode and self._parameterNode.reconstructedVolume:
            self.logic.setVolumeRenderingProperty(self._parameterNode.reconstructedVolume, window=200, level=(255 - value))
    
    def onSetRoiButton(self) -> None:
        """
        Set volume reconstruction ROI and ReferenceToRas transform based on the current location of the ultrasound image.
        The center of ultrasound will be the center of the ROI. Marked (X) direction of the image will be aligne to Right (R) and Far (Y) to Anterior (A).
        """
        self.logic.resetReferenceToRasBasedOnImage()
        self.logic.resetRoiBasedOnImage()
        
#
# NeedleGuideLogic
#


class NeedleGuideLogic(ScriptedLoadableModuleLogic):
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
        
        self.reconstructing = False

    def getParameterNode(self):
        return NeedleGuideParameterNode(super().getParameterNode())
    
    def startVolumeReconstruction(self):
        """
        Start live volume reconstruction.
        """
        parameterNode = self.getParameterNode()
        self.reconstructing = True
        reconstructionLogic = slicer.modules.volumereconstruction.logic()
        reconstructionLogic.StartLiveVolumeReconstruction(parameterNode.reconstructorNode)
        outputVolume = parameterNode.reconstructorNode.GetOutputVolumeNode()
        self.setVolumeRenderingProperty(outputVolume, window=200, level=(255-parameterNode.opacityThreshold))
        parameterNode.reconstructedVolume = outputVolume
    
    def stopVolumeReconstruction(self):
        """
        Stop live volume reconstruction.
        """
        parameterNode = self.getParameterNode()
        self.reconstructing = False
        reconstructionLogic = slicer.modules.volumereconstruction.logic()
        reconstructionLogic.StopLiveVolumeReconstruction(parameterNode.reconstructorNode)
    
    def setVolumeRenderingProperty(self, volumeNode, window=255, level=127):
        volumeRenderingLogic = slicer.modules.volumerendering.logic()
        volumeRenderingDisplayNode = volumeRenderingLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if not volumeRenderingDisplayNode:
            volumeRenderingDisplayNode = volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
            
        upper = min(255 + window, level + window/2)
        lower = max(0 - window, level - window/2)

        if upper <= lower:
            upper = lower + 1  # Make sure the displayed intensity range is valid.

        p0 = lower
        p1 = lower + (upper - lower)*0.15
        p2 = lower + (upper - lower)*0.4
        p3 = upper

        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(p0, 0.0)
        opacityTransferFunction.AddPoint(p1, 0.2)
        opacityTransferFunction.AddPoint(p2, 0.6)
        opacityTransferFunction.AddPoint(p3, 1)

        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(p0, 0.20, 0.10, 0.00)
        colorTransferFunction.AddRGBPoint(p1, 0.65, 0.45, 0.15)
        colorTransferFunction.AddRGBPoint(p2, 0.85, 0.75, 0.55)
        colorTransferFunction.AddRGBPoint(p3, 1.00, 1.00, 0.80)

        volumeProperty = volumeRenderingDisplayNode.GetVolumePropertyNode().GetVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()    
    
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
    
    def resetReferenceToRasBasedOnImage(self):
        """
        Get the current position of Image in RAS. Make sure ReferenceToRas transform is aligned with the image.
        Image should be aligned so X is Right, and Y is Anterior.
        """
        parameterNode = self.getParameterNode()
        
        inputVolume = parameterNode.inputVolume
        if not inputVolume:
            logging.error("Input volume is not set")
            return
        
        referenceToRas = parameterNode.referenceToRas
        if not referenceToRas:
            logging.error("ReferenceToRas transform is not set")
            return
        
        # Temporarily set ReferenceToRas matrix to identity
        referenceToRasMatrix = vtk.vtkMatrix4x4()
        referenceToRas.GetMatrixTransformToWorld(referenceToRasMatrix)
        referenceToRasMatrix.Identity()
        referenceToRas.SetMatrixTransformToParent(referenceToRasMatrix)
        
        # Get the current position of Image in RAS
        imageToReferenceTransform = slicer.mrmlScene.GetNodeByID(inputVolume.GetTransformNodeID())
        if imageToReferenceTransform is None:
            logging.error("Image transform is not set")
            return
        
        imageToReferenceMatrix = vtk.vtkMatrix4x4()
        imageToReferenceTransform.GetMatrixTransformToWorld(imageToReferenceMatrix)
        imageToReferenceMatrix.Invert()
        # Keep only rotation part from imageToReferenceMatrix to align ReferenceToRas with the image
        referenceToImageTransform = vtk.vtkTransform()
        referenceToImageTransform.SetMatrix(imageToReferenceMatrix)
        referenceToRasTransform = vtk.vtkTransform()
        wxyz = referenceToImageTransform.GetOrientationWXYZ()
        referenceToRasTransform.RotateWXYZ(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
        referenceToRas.SetMatrixTransformToParent(referenceToRasTransform.GetMatrix())
    
    def resetRoiBasedOnImage(self):
        """
        Get the current position of Image in RAS. Make sure volume reconstruction has a ROI node and it is centered in the image.
        """
        parameterNode = self.getParameterNode()
        if not parameterNode.reconstructorNode:
            logging.error("Reconstructor node is not set")
            return
        
        # Get the current position of Image in RAS
        imageNode = parameterNode.inputVolume
        if not imageNode:
            logging.warning("Cannot set ROI because input volume is not set")
            return
        
        # Get the center of the image
        imageBounds_Ras = np.zeros(6)
        imageNode.GetRASBounds(imageBounds_Ras)
        imageCenter_Ras = np.zeros(3)
        for i in range(3):
            imageCenter_Ras[i] = (imageBounds_Ras[i*2] + imageBounds_Ras[i*2+1]) / 2
        
        # Check if volume reconstruction node has a ROI already
        roiNode = parameterNode.reconstructorNode.GetInputROINode()
        if not roiNode:
            roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
            parameterNode.reconstructorNode.SetAndObserveInputROINode(roiNode)
        roiNode.SetCenterWorld(imageCenter_Ras)

#
# NeedleGuideTest
#


class NeedleGuideTest(ScriptedLoadableModuleTest):
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
        self.test_NeedleGuide1()

    def test_NeedleGuide1(self):
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
        inputVolume = SampleData.downloadSample("NeedleGuide1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = NeedleGuideLogic()

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
