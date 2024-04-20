
from collections import defaultdict
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import time
import numpy as np

from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.encaps import encapsulate
from pydicom.uid import ImplicitVRLittleEndian, generate_uid, JPEGBaseline
from pydicom.multival import MultiValue
from DICOMLib import DICOMUtils
import qt
import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from DICOMLib import DICOMUtils

try:
    import cv2
except ImportError:
    slicer.util.pip_install('opencv-python')
    import cv2

#
# AnonymizeUltrasound
#

class AnonymizeUltrasound(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Anonymize ultrasound"
        self.parent.categories = ["Ultrasound"]
        self.parent.dependencies = []
        self.parent.contributors = ["Tamas Ungi (Queen's University)"]
        self.parent.helpText = """
This is a module for anonymizing ultrasound images and sequences stored in DICOM folders.
The mask (green contour) signals what part of the image will stay after applying the mask. The image area under the green contour will be kept along with the pixels inside the contour.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", performPostModuleDiscoveryTasks)

#
# Perform module initialization after the application has started
#

def performPostModuleDiscoveryTasks():
    """
    Perform some initialization tasks that require the application to be fully started up.
    """
    try:
        import pandas as pd
    except ImportError:
        slicer.util.pip_install('pandas')
        import pandas as pd
    
#
# AnonymizeUltrasoundWidget
#

class AnonymizeUltrasoundWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.compositingModeExit = 0
        self.notEnteredYet = True

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/AnonymizeUltrasound.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AnonymizeUltrasoundLogic()
        
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputDirectoryButton.directoryChanged.connect(self.updateInputDirectoryFromWidget)
        self.ui.outputDirectoryButton.directoryChanged.connect(self.updateOutputDirectoryFromWidget)
        self.ui.skipSingleframeCheckBox.connect('toggled(bool)', self.updateSettingsFromGUI)
        self.ui.continueProgressCheckBox.connect('toggled(bool)', self.updateSettingsFromGUI)
        self.ui.keepFoldersCheckBox.connect('toggled(bool)', self.updateSettingsFromGUI)
        self.ui.convertGrayscaleCheckBox.connect('toggled(bool)', self.updateSettingsFromGUI)
        self.ui.compressionCheckBox.connect('toggled(bool)', self.updateSettingsFromGUI)
        
        # Buttons
        self.ui.updateDicomsButton.connect('clicked(bool)', self.onUpdateDicomsButton)
        self.ui.nextSeriesButton.connect('clicked(bool)', self.onNextSeriesButton)
        self.ui.maskLandmarksButton.connect('toggled(bool)', self.onMaskLandmarksButton)
        self.ui.exportScanButton.connect('clicked(bool)', self.onExportScanButton)

        self.ui.labelsFileSelector.connect('currentPathChanged(QString)', self.onLabelsPathChanged)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
    def updateSettingsFromGUI(self):
        settings = qt.QSettings()

        if self.ui.skipSingleframeCheckBox.checked:
            settings.setValue('AnonymizeUltrasound/SkipSingleframe', "True")
        else:
            settings.setValue('AnonymizeUltrasound/SkipSingleframe', "False")
        
        if self.ui.continueProgressCheckBox.checked:
            settings.setValue('AnonymizeUltrasound/ContinueProgress', "True")
        else:
            settings.setValue('AnonymizeUltrasound/ContinueProgress', "False")
        
        if self.ui.compressionCheckBox.checked:
            settings.setValue('AnonymizeUltrasound/Compression', "True")
        else:
            settings.setValue('AnonymizeUltrasound/Compression', "False")
        
        if self.ui.convertGrayscaleCheckBox.checked:
            settings.setValue('AnonymizeUltrasound/ConvertGrayscale', "True")
        else:
            settings.setValue('AnonymizeUltrasound/ConvertGrayscale', "False")

        if self.ui.keepFoldersCheckBox.checked:
            settings.setValue('AnonymizeUltrasound/KeepFolderStructure', "True")
        else:
            settings.setValue('AnonymizeUltrasound/keepFolderStructure', "False")

    def onLabelsPathChanged(self, filePath):
        settings = qt.QSettings()
        settings.setValue('AnonymizeUltrasound/LabelsPath', filePath)

        categories = defaultdict(list)

        try:
            with open(filePath, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    category, label = map(str.strip, row)
                    categories[category].append(label)
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Cannot read labels file: {filePath}, error: {e}")

        # Remove all existing labels from the scroll area
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())): 
            # self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget().setParent(None)
            self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget().deleteLater()

        # Populate labels scroll area
        for category, labels in categories.items():
            categoryGroupBox = qt.QGroupBox(category, self.ui.labelsScrollAreaWidgetContents)
            categoryLayout = qt.QVBoxLayout(categoryGroupBox)
            for label in labels:
                checkBox = qt.QCheckBox(label, categoryGroupBox)
                categoryLayout.addWidget(checkBox)
            categoryGroupBox.setLayout(categoryLayout)
            self.ui.labelsScrollAreaWidgetContents.layout().addWidget(categoryGroupBox)
        
        self.ui.labelsScrollAreaWidgetContents.layout().addStretch(1)
    
    def updateInputDirectoryFromWidget(self):
        """
        Called when user changes the directory in the directory browser.
        """
        # Save current directory in application settings, so it is remembered when the module is re-opened

        directory = self.ui.inputDirectoryButton.directory
        settings = qt.QSettings()
        settings.setValue('AnonymizeUltrasound/InputDirectory', directory)

    def updateOutputDirectoryFromWidget(self):
        """
        Called when user changes the directory in the directory browser.
        """
        # Save current directory in application settings, so it is remembered when the module is re-opened

        directory = self.ui.outputDirectoryButton.directory
        settings = qt.QSettings()
        settings.setValue('AnonymizeUltrasound/OutputDirectory', directory)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def initializeGui(self):
        moduleWidget = slicer.modules.anonymizeultrasound.widgetRepresentation().self()
        settings = qt.QSettings()

        # Set default labels file path
        labelsPath = settings.value('AnonymizeUltrasound/LabelsPath', '')
        
        # If labels file does not exist, reset to ''
        if not os.path.exists(labelsPath):
            labelsPath = ''

        # If labels file is not set, use the default labels file
        if labelsPath == '':
            labelsPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Resources/default_labels.csv')
        moduleWidget.ui.labelsFileSelector.currentPath = labelsPath

        directory = settings.value('AnonymizeUltrasound/InputDirectory', '')
        moduleWidget.ui.inputDirectoryButton.directory = directory

        directory = settings.value('AnonymizeUltrasound/OutputDirectory', '')
        moduleWidget.ui.outputDirectoryButton.directory = directory

        # Set default settings values if they are missing from settings file
        settingNames =  ["SkipSingleFrame",
                         "ContinueProgress",
                         "KeepFolderStructure",
                         "ConvertGrayscale",
                         "Compression"]
        defaultValues = ["True",
                         "True",
                         "False",
                         "False",
                         "True"]
        for settingName, defaultValue in zip(settingNames, defaultValues):
            if settings.value(f'AnonymizeUltrasound/{settingName}', '') == '':
                settings.setValue(f'AnonymizeUltrasound/{settingName}', defaultValue)
        
        # Make sure length/precision setting is at least 3
        if int(settings.value('length/precision', '0')) < 3:
            settings.setValue('length/precision', "3")
            applicationLength = slicer.mrmlScene.GetNodeByID("vtkMRMLUnitNodeApplicationLength")
            if applicationLength is None:
                logging.error("Cannot set length precision because node ID not found: vtkMRMLUnitNodeApplicationLength")
            else:
                applicationLength.SetPrecision(3)

        # Set GUI values from settings
        skipSingleFrame = settings.value('AnonymizeUltrasound/SkipSingleframe', "True") == "True"
        continueProgress = settings.value('AnonymizeUltrasound/ContinueProgress', "True") == "True"
        keepFolders = settings.value('AnonymizeUltrasound/KeepFolderStructure', "True") == "True"
        convertGrayscale = settings.value('AnonymizeUltrasound/ConvertGrayscale', "True") == "True"
        compression = settings.value('AnonymizeUltrasound/Compression', "True") == "True"
        moduleWidget.ui.skipSingleframeCheckBox.checked = skipSingleFrame
        moduleWidget.ui.continueProgressCheckBox.checked = continueProgress
        moduleWidget.ui.keepFoldersCheckBox.checked = keepFolders
        moduleWidget.ui.convertGrayscaleCheckBox.checked = convertGrayscale
        moduleWidget.ui.compressionCheckBox.checked = compression

        # Set initial states of widgets
        self.ui.inputsCollapsibleButton.collapsed = False
        self.ui.dataProcessingCollapsibleButton.collapsed = True
        self.ui.labelsCollapsibleButton.collapsed = True
        self.ui.settingsCollapsibleButton.collapsed = True
        
    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

        # Collapse DataProbe widget
        mw = slicer.util.mainWindow()
        if mw:
            w = slicer.util.findChild(mw, "DataProbeCollapsibleWidget")
            if w:
                w.collapsed = True

        # If this is the first enter, initialize GUI
        if self.notEnteredYet:
            self.notEnteredYet = False
            self.initializeGui()
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        self.compositingModeExit = sliceCompositeNode.GetCompositing()
        sliceCompositeNode.SetCompositing(2)

        # Make sure all nodes exist
        self.logic.setupScene()


    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromData)
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        sliceCompositeNode.SetCompositing(self.compositingModeExit)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromData):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromData)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromData)

        # Initial GUI update
        self.updateGUIFromData()

    def updateGUIFromData(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update widgets

        if self.logic.currentDicomDataset:
            currentPatientId = self.logic.currentDicomDataset.PatientName
            if currentPatientId:
                self.ui.currentPatientLabel.text = currentPatientId
            else:
                self.ui.currentPatientLabel.text = 'None'

            currentStudyId = self.logic.currentDicomDataset.StudyID
            if currentStudyId:
                self.ui.currentStudyLabel.text = currentStudyId
            else:
                self.ui.currentStudyLabel.text = 'None'

            currentSeriesId = self.logic.currentDicomDataset.SeriesNumber
            if currentSeriesId:
                self.ui.currentSeriesLabel.text = currentSeriesId
            else:
                self.ui.currentSeriesLabel.text = 'None'

        status = self._parameterNode.GetParameter(self.logic.STATUS)
        if status == self.logic.STATUS_LANDMARK_PLACEMENT:
            # Set mouse interaction mode to landmark placement
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
            self.ui.maskLandmarksButton.checked = True
            self.ui.statusLabel.text = "Click of four corners of the ultrasound image area"
        elif status == self.logic.STATUS_LANDMARKS_PLACED:
            # Set mouse interaction mode to default
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            self.ui.maskLandmarksButton.checked = False
            self.ui.statusLabel.text = "Click Export to save DICOM file in output folder"

        # Update buttons states and tooltips
        
        self.ui.updateDicomsButton.enabled = True

        numInstances = self.logic.getNumberOfInstances()
        if numInstances > 1:
            self.ui.progressBar.maximum = numInstances
            self.ui.dataProcessingCollapsibleButton.enabled = True
            self.ui.labelsCollapsibleButton.enabled = True
        else:
            self.ui.dataProcessingCollapsibleButton.enabled = False
            self.ui.labelsCollapsibleButton.enabled = False
            self.ui.statusLabel.text = "Select input folder and press Read DICOM folder button to load DICOM files"

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onUpdateDicomsButton(self):
        """
        Run processing when user clicks "Read DICOM folder" button.
        """
        logging.info('onUpdateDicomsButton')

        # Check if input directory is specified
        inputDirectory = self.ui.inputDirectoryButton.directory
        if not inputDirectory:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "AnonymizeUltrasound", "Input directory must be specified")
            return
            
        # Check if the specified input directory exists. The folder can be renamed after selecting it, so in theory this is possible.
        if not os.path.exists(inputDirectory):
            qt.QMessageBox.critical(slicer.util.mainWindow(), "AnonymizeUltrasound", "Input directory does not exist")
            return
        
        # Check if output directory is specified
        outputDirectory = self.ui.outputDirectoryButton.directory
        if not outputDirectory:
            qt.QMessageBox.critical(slicer.util.mainWindow(), "AnonymizeUltrasound", "Output directory must be specified")
            return
        
        # Check if the specified output directory exists. The folder can be renamed after selecting it, so in theory this is possible.
        if not os.path.exists(outputDirectory):
            qt.QMessageBox.critical(slicer.util.mainWindow(), "AnonymizeUltrasound", "Output directory does not exist")
            return
        
        # Find all DICOM files in the input directory
        numFiles = self.logic.updateDicomDf(inputDirectory, self.ui.skipSingleframeCheckBox.checked)
        
        # Export self.logic.dicomDf as a CSV file in the outputDirectory to make sure original keys to anonymized IDs are saved
        outputDirectory = self.ui.outputDirectoryButton.directory
        os.makedirs(outputDirectory, exist_ok=True)  # Make sure output directory exists
        outputFilePath = os.path.join(outputDirectory, "keys.csv")
        self.logic.dicomDf.to_csv(outputFilePath, index=False)
        
        statusText = str(numFiles)
        if self.ui.skipSingleframeCheckBox.checked:
            statusText += " multi-frame dicom files found in input folder."
        else:
            statusText += " dicom files found in input folder."
        
        if self.ui.continueProgressCheckBox.checked:
            numDone = self.logic.updateProgressDicomDf(inputDirectory, outputDirectory, self.ui.keepFoldersCheckBox.checked)
            if numDone < 1:
                statusText += '\nNo files already processed. Starting from first in alphabetical order.'
            else:
                statusText += '\n' + str(numDone) + ' files already processed in output folder. Continue at next.'
        self.ui.statusLabel.text = statusText

        # Update widgets to represent loaded data
        self.updateGUIFromData()

        if numFiles > 1:
            self.ui.inputsCollapsibleButton.collapsed = True
            self.ui.dataProcessingCollapsibleButton.collapsed = False
            self.ui.labelsCollapsibleButton.collapsed = False
        else:
            self.ui.inputsCollapsibleButton.collapsed = False
            self.ui.dataProcessingCollapsibleButton.collapsed = True
            self.ui.labelsCollapsibleButton.collapsed = True

    def onNextSeriesButton(self):
        logging.info('Next study button pressed')

        dialog = self.createWaitDialog("Loading series", "Please wait until the DICOM instance is loaded...")

        # Load the next series

        currentDicomDfIndex = self.logic.loadNextSequence()
        if currentDicomDfIndex is None:
            statusText = "No more series to load"
            self.ui.statusLabel.text = statusText
            return
        else:
            self.ui.progressBar.value = currentDicomDfIndex

        dialog.close()

        self.logic.updateMaskVolume()
        
        # Uncheck all label checkboxes
        for i in range(self.ui.labelsScrollAreaWidgetContents.layout().count()): 
            groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            if groupBox is None:
                continue
            # Find all checkboxes in groupBox
            for j in range(groupBox.layout().count()): 
                checkBox = groupBox.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox):
                    checkBox.setChecked(False)

        # Update GUI

        patientUID = self.logic.currentDicomDataset.PatientName
        if patientUID:
            self.ui.currentPatientLabel.text = patientUID
        else:
            self.ui.currentPatientLabel.text = 'None'

        studyID = self.logic.currentDicomDataset.StudyID
        if studyID:
            self.ui.currentStudyLabel.text = studyID
        else:
            self.ui.currentStudyLabel.text = 'None'

        seriesID = self.logic.currentDicomDataset.SeriesInstanceUID
        if seriesID:
            self.ui.currentSeriesLabel.text = seriesID
        else:
            self.ui.currentSeriesLabel.text = 'None'

        instanceUID = self.logic.currentDicomDataset.SOPInstanceUID
        if instanceUID is None:
            self.ui.currentInstanceLabel.text = 'None'

        statusText = f"Instance {instanceUID} loaded from file:\n"

        # Get the file path from the dataframe
        
        filepath = self.logic.dicomDf.iloc[currentDicomDfIndex].Filepath
        statusText += filepath
        self.ui.statusLabel.text = statusText
        
        self.logic.showMaskContour()

        # Set red slice compositing mode to 2
        sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        sliceCompositeNode.SetCompositing(2)

    def onMaskLandmarksButton(self, toggled):
        logging.info('Mask landmarks button pressed')

        landmarkNode = self._parameterNode.GetNodeReference(self.logic.MASK_FAN_LANDMARKS)
        if landmarkNode is None:
            logging.error(f"Landmark node not found: {self.logic.MASK_FAN_LANDMARKS}")
            return

        if toggled:
            self.logic.resetMaskLandmarks()
            self._parameterNode.SetParameter(self.logic.STATUS, self.logic.STATUS_LANDMARK_PLACEMENT)
            landmarkNodeId = landmarkNode.GetID()
            landmarkNode.SetDisplayVisibility(True)

            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            if landmarkNode:
                selectionNode.SetReferenceActivePlaceNodeClassName(landmarkNode.GetClassName())
            selectionNode.SetReferenceActivePlaceNodeID(landmarkNodeId)

            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SwitchToPersistentPlaceMode()
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        
    def createWaitDialog(self, title, message):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        dialogLayout = qt.QVBoxLayout(dialog)
        dialogLayout.setContentsMargins(20, 14, 20, 14)
        dialogLayout.setSpacing(4)
        dialogLayout.addStretch(1)
        dialogLabel = qt.QLabel(message)
        dialogLabel.setAlignment(qt.Qt.AlignCenter)
        dialogLayout.addWidget(dialogLabel)
        dialogLayout.addStretch(1)
        dialog.show()
        slicer.app.processEvents()

        return dialog

    def onExportScanButton(self):
        """
        Callback function for the export scan button.
        """
        logging.info('Export scan button pressed')

        convertToGrayscale = self.ui.convertGrayscaleCheckBox.checked

        # Create modal dialog asking the user to wait until the export is complete. Use QDialog instead of QProgressDialog

        # Save current frame index for sequence
        currentSequenceBrowser = self._parameterNode.GetNodeReference(self.logic.CURRENT_SEQUENCE)
        if currentSequenceBrowser is None:
            self.ui.statusLabel.text = "Load a DICOM sequence before trying to export"
            return
        selectedItemNumber = currentSequenceBrowser.GetSelectedItemNumber()
        
        dialog = self.createWaitDialog("Exporting scan", "Please wait until the scan is exported...")
        
        # Check if any labels are checked
        annotationLabels = []
        for i in reversed(range(self.ui.labelsScrollAreaWidgetContents.layout().count())): 
            groupBox = self.ui.labelsScrollAreaWidgetContents.layout().itemAt(i).widget()
            if groupBox is None:
                continue
            # Find all checkboxes in groupBox
            for j in reversed(range(groupBox.layout().count())):
                checkBox = groupBox.layout().itemAt(j).widget()
                if isinstance(checkBox, qt.QCheckBox) and checkBox.isChecked():
                    annotationLabels.append(checkBox.text)
        
        # Mask images
        self.logic.maskSequence()
        
        # Check setting if original folder and filename should be kept
        inputDirectory = self.ui.inputDirectoryButton.directory
        outputDirectory = self.ui.outputDirectoryButton.directory
        if self.ui.keepFoldersCheckBox.checked:
            outputRelativePath, filename = self.logic.currentRelativePath(inputDicomFolder=inputDirectory)
            if not filename.endswith(".dcm"):
                filename += ".dcm"
            outputPath = os.path.join(outputDirectory, outputRelativePath)
        else:
            outputPath = outputDirectory
            filename, _, _ = self.logic.generateNameFromDicomData(self.logic.currentDicomDataset)
            
        # Export the scan
        dicomFilePath, jsonFilePath, dicomHeaderFilePath = self.logic.exportDicom(
            outputDirectory=outputPath,
            outputFilename=filename,
            convertToGrayscale=convertToGrayscale,
            labels = annotationLabels,
            compression=self.ui.compressionCheckBox.checked)
        
        # Restore selected item number in sequence browser
        currentSequenceBrowser.SetSelectedItemNumber(selectedItemNumber)
        
        # Display file paths in the status label

        statusText = "DICOM saved to: " + dicomFilePath + "\nAnnotations saved to: " + jsonFilePath\
                        + "\nDICOM header saved to: " + dicomHeaderFilePath

        self.ui.statusLabel.text = statusText

        # Close the modal dialog

        dialog.close()

# AnonymizeUltrasoundLogic
#

class AnonymizeUltrasoundLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    CURRENT_PATIENT_ID = "CurrentPatientID"
    CURRENT_STUDY_ID = "CurrentStudyID"
    CURRENT_SERIES_ID = "CurrentSeriesID"
    CURRENT_SEQUENCE = "CurrentSequence"

    MASK_FAN_LANDMARKS = "MaskFanLandmarks"
    MARK_RECTANGLE_LANDMARKS = "MarkRectangleLandmarks"
    MASK_VOLUME = "MaskVolume"
    MASK_CONTOUR_VOLUME = "MaskContourVolume"

    STATUS = "Status"
    STATUS_INITIAL = "StatusInitial"  # When the application is just started, no data parsed or loaded yet
    STATUS_PATIENT_LOADED = "StatusPatientloaded"
    STATUS_LANDMARK_PLACEMENT = "StatusLandmarkPlacement"
    STATUS_LANDMARKS_PLACED = "StatusLandmarksPlaced"
    STATUS_MASKING = "StatusMasking"

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)

        self.loadedSequenceBrowserNodeIds = []
        self.currentDicomHeader = None
        self.currentDicomDataset = None
        self.maskParameters = {}
        self.seriesQueue = []
        self.currentSeries = None
        self.dicomDf = None
        self.nextDicomDfIndex = 0

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter(self.CURRENT_PATIENT_ID):
            parameterNode.SetParameter(self.CURRENT_PATIENT_ID, "")
        if not parameterNode.GetParameter(self.CURRENT_STUDY_ID):
            parameterNode.SetParameter(self.CURRENT_STUDY_ID, "")
        if not parameterNode.GetParameter(self.CURRENT_SERIES_ID):
            parameterNode.SetParameter(self.CURRENT_SERIES_ID, "")
        if not parameterNode.GetParameter(self.CURRENT_SEQUENCE):
            parameterNode.SetParameter(self.CURRENT_SEQUENCE, "")
        if not parameterNode.GetParameter(self.STATUS):
            parameterNode.SetParameter(self.STATUS, self.STATUS_INITIAL)

    def setupScene(self, maskControlPointsList=None):

        # Make sure fan mask markups fudicual node exists and referenced by the parameter node

        parameterNode = self.getParameterNode()
        markupsNode = parameterNode.GetNodeReference(self.MASK_FAN_LANDMARKS)
        if not markupsNode:
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            markupsNode.SetName(self.MASK_FAN_LANDMARKS)
            markupsNode.GetDisplayNode().SetTextScale(0.0)
            parameterNode.SetNodeReferenceID(self.MASK_FAN_LANDMARKS, markupsNode.GetID())
        
        if maskControlPointsList:
            markupsNode.RemoveAllControlPoints()
            for controlPoint in maskControlPointsList:
                markupsNode.AddControlPoint(controlPoint)

        self.addObserver(markupsNode, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onPointAdded)
        self.addObserver(markupsNode, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onPointDefined)
        self.addObserver(markupsNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onPointModified)

        # Add observer for node added to mrmlScene
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onNodeAdded)

        # Make sure rectangle mask markups fudicual node exists and referenced by the parameter node

        if not parameterNode.GetNodeReference(self.MARK_RECTANGLE_LANDMARKS):
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            markupsNode.SetName(self.MARK_RECTANGLE_LANDMARKS)
            parameterNode.SetNodeReferenceID(self.MARK_RECTANGLE_LANDMARKS, markupsNode.GetID())

    def onNodeAdded(self, caller, event):
        """
        Called when a node is added to the scene.
        """
        node = caller
        if node.IsA("vtkMRMLSequenceBrowserNode"):
            logging.info(f"Sequence browser node added: {node.GetID()}")
            # Set newly added sequence browser node as current sequence browser node
            parameterNode = self.getParameterNode()
            parameterNode.SetNodeReferenceID(self.CURRENT_SEQUENCE, node.GetID())

    def updateDicomDf(self, input_folder, skip_singleframe=False):
        """
        Update dicomDf with a list of all DICOM files in the input folder.
        """
        dicom_data = []
    
        # Get the total number of files
        total_files = sum([len(files) for root, dirs, files in os.walk(input_folder)])

        # Create a QProgressDialog
        progress_dialog = qt.QProgressDialog("Parsing DICOM files...", "Cancel", 0, total_files, slicer.util.mainWindow())
        progress_dialog.setWindowModality(qt.Qt.WindowModal)
        progress_dialog.show()
        slicer.app.processEvents()

        # Recursively walk through the input folder
        file_count = 0
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                progress_dialog.setValue(file_count)
                file_count += 1
                slicer.app.processEvents()
                
                # Construct the full file path
                file_path = os.path.join(root, file)

                try:
                    # Try to read the file as a DICOM file
                    dicom_ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    
                    # Try to get image spacing.
                    physical_delta_x = None
                    physical_delta_y = None
                    to_patch = True
                    if hasattr(dicom_ds, "SequenceOfUltrasoundRegions"):
                        if len(dicom_ds.SequenceOfUltrasoundRegions) > 0:
                            region = dicom_ds.SequenceOfUltrasoundRegions[0]
                            if (hasattr(region, "PhysicalDeltaX") and hasattr(region, "PhysicalDeltaY")):
                                physical_delta_x = region.PhysicalDeltaX
                                physical_delta_y = region.PhysicalDeltaY
                                to_patch = False
                    
                    # Extract required information
                    patient_uid = dicom_ds.PatientID if 'PatientID' in dicom_ds else None
                    study_uid = dicom_ds.StudyInstanceUID if 'StudyInstanceUID' in dicom_ds else None
                    series_uid = dicom_ds.SeriesInstanceUID if 'SeriesInstanceUID' in dicom_ds else None
                    instance_uid = dicom_ds.SOPInstanceUID if 'SOPInstanceUID' in dicom_ds else None
                    
                    exp_filename, _, _ = self.generateNameFromDicomData(dicom_ds)
                    
                    if skip_singleframe and ('NumberOfFrames' not in dicom_ds or dicom_ds.NumberOfFrames < 2):
                        continue

                    # Append the information to the list, if PatientID, StudyInstanceUID, and SeriesInstanceUID are present
                    if patient_uid and study_uid and series_uid and instance_uid:
                        dicom_data.append([file_path, exp_filename, patient_uid, study_uid, series_uid, instance_uid, physical_delta_x, physical_delta_y, to_patch])
                except Exception as e:
                    # If the file is not a valid DICOM file, continue to the next file
                    continue

        # Update dicomDf
        self.dicomDf = pd.DataFrame(dicom_data, columns=['Filepath', 'AnonFilename', 'PatientUID', 'StudyUID', 'SeriesUID', 'InstanceUID', 'PhysicalDeltaX', 'PhysicalDeltaY', 'Patch'])
        self.dicomDf = self.dicomDf.sort_values(by='Filepath')  # This makes a difference on Mac, not on Windows.
        
        # This is a workaround for the issue that some DICOM files do not have spacing information. The information may be used when loading each file,
        # but patching the DICOM files before importing them would be a better option.
        self.dicomDf['PhysicalDeltaX'] = self.dicomDf.groupby('StudyUID')['PhysicalDeltaX'].transform(lambda x: x.ffill().bfill())
        self.dicomDf['PhysicalDeltaY'] = self.dicomDf.groupby('StudyUID')['PhysicalDeltaY'].transform(lambda x: x.ffill().bfill())
        
        # If PhysicalDeltaX is still missing from at least one row, log a warning.
        if self.dicomDf['PhysicalDeltaX'].isnull().sum() > 0:
            logging.warning("Some ultrasound scans have missing spacing information.")
        
        self.nextDicomDfIndex = 0

        # Close the progress dialog
        progress_dialog.setValue(total_files)
        progress_dialog.close()

        # Return the number of rows in the dataframe
        return len(self.dicomDf)
    
    def updateProgressDicomDf(self, input_folder, output_folder, keep_folders=False):
        """
        Check the output folder to see what input files are already processed.
        
        :param input_folder: full path to the input folder where input DCM files are.
        :param output_folder: full path to the output folder where already processed files can be found.
        :param keep_folders: If True, output files are expected by the same name in the same subfolders as input files.
        :return:  index for dicomDf that points to the next row that needs to be processed.
        """
        # If there is no input parsed, no need to check output
        numSeries = self.getNumberOfSeries()
        if numSeries < 1:
            return 0
        
        # Iterate through the input files based on self.dicomDf and check if the expected output file exists.
        for index, row in self.dicomDf.iterrows():
            input_file = row['Filepath']
            
            if keep_folders:
                output_fullpath = os.path.join(output_folder, os.path.relpath(input_file, input_folder))
            else:
                output_path = output_folder
                output_filename = row['AnonFilename']
                output_fullpath = os.path.join(output_path, output_filename)
            
            if not os.path.exists(output_fullpath):
                self.nextDicomDfIndex = index
                return index
            else:
                # Output file already exists, skip processing
                continue
    
    def getNumberOfSeries(self):
        """
        Return the number of series in the current DICOM dataframe.
        """
        return self.dicomDf.SeriesUID.nunique()
    
    def getNumberOfInstances(self):
        """
        Return the number of instances in the current DICOM dataframe.
        """
        if self.dicomDf is None:
            return 0
        else:
            return len(self.dicomDf)

    def removeMarkupObservers(self):
        self.removeObservers(self.onPointAdded)
        self.removeObservers(self.onPointDefined)
        self.removeObservers(self.onPointModified)

    def resetMaskLandmarks(self):
        parameterNode = self.getParameterNode()
        markupsNode = parameterNode.GetNodeReference(self.MASK_FAN_LANDMARKS)
        if markupsNode:
            markupsNode.RemoveAllControlPoints()

    def onPointAdded(self, caller=None, event=None):
        logging.info('Point added')
        parameterNode = self.getParameterNode()

        markupsNode = parameterNode.GetNodeReference(self.MASK_FAN_LANDMARKS)
        if not markupsNode:
            logging.error("Markups node not found")
            return

        if markupsNode.GetNumberOfControlPoints() > 3:
            self.updateMaskVolume()
            self.showMaskContour()
        else:
            slicer.util.setSliceViewerLayers(foreground=None)

    def showMaskContour(self, show=True):
        parameterNode = self.getParameterNode()
        maskContourVolumeNode = parameterNode.GetNodeReference(self.MASK_CONTOUR_VOLUME)

        # Set mask contour as foreground volume in slice viewers, and make sure it has 50% opacity
        if maskContourVolumeNode and show:
            slicer.util.setSliceViewerLayers(foreground=maskContourVolumeNode)
            sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
            sliceCompositeNode.SetForegroundOpacity(0.5)
            # Make sure background and foreground are not alpha blended but added
            sliceCompositeNode.SetCompositing(2) # Add foreground and background
            # Set window and level so that mask is visible
            displayNode = maskContourVolumeNode.GetDisplayNode()
            displayNode.SetWindow(1)
            displayNode.SetLevel(0.5)
        else:
            slicer.util.setSliceViewerLayers(foreground=None)

    def onPointModified(self, caller=None, event=None):
        parameterNode = self.getParameterNode()

        markupsNode = parameterNode.GetNodeReference(self.MASK_FAN_LANDMARKS)
        if not markupsNode:
            logging.error("Markups node not found")
            return

        if markupsNode.GetNumberOfControlPoints() > 3:
            self.updateMaskVolume()
            maskContourVolumeNode = parameterNode.GetNodeReference(self.MASK_CONTOUR_VOLUME)
            if maskContourVolumeNode:
                sliceCompositeNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
                if sliceCompositeNode.GetForegroundVolumeID() != maskContourVolumeNode.GetID():
                    sliceCompositeNode.SetForegroundVolumeID(maskContourVolumeNode.GetID())
                    sliceCompositeNode.SetForegroundOpacity(0.5)
                    displayNode = maskContourVolumeNode.GetDisplayNode()
                    displayNode.SetWindow(1)
                    displayNode.SetLevel(0.5)


    def onPointDefined(self, caller=None, event=None):
        logging.info('Point defined')
        parameterNode = self.getParameterNode()

        markupsNode = parameterNode.GetNodeReference(self.MASK_FAN_LANDMARKS)
        if not markupsNode:
            logging.error("Markups node not found")
            return

        if markupsNode.GetNumberOfControlPoints() > 2:
            self.updateMaskVolume()

        if markupsNode.GetNumberOfControlPoints() > 3:
            parameterNode.SetParameter(self.STATUS, self.STATUS_LANDMARKS_PLACED)
        else:
            parameterNode.SetParameter(self.STATUS, self.STATUS_LANDMARK_PLACEMENT)

    def getCurrentPatientName(self):
        if self.currentDicomDataset is None:
            logging.error("Current DICOM dataset is not set, cannot return patient name")
            return None
        else:
            return self.currentDicomDataset.PatientName

    def getCurrentProxyNode(self):
        """
        Get the proxy node of the master sequence node of the currently selected sequence browser node
        """
        parameterNode = self.getParameterNode()
        currentBrowserNode = parameterNode.GetNodeReference(self.CURRENT_SEQUENCE)
        if currentBrowserNode is None:
            logging.error("Current sequence browser node not found using reference: " + self.CURRENT_SEQUENCE)
            return None

        # Get the proxy node of the master sequence node of the selected sequence browser node

        masterSequenceNode = currentBrowserNode.GetMasterSequenceNode()
        if masterSequenceNode is None:
            logging.error("Master sequence node missing for browser node with ID " + currentBrowserNode.GetID())
            return None

        proxyNode = currentBrowserNode.GetProxyNode(masterSequenceNode)
        if proxyNode is None:
            logging.error("Proxy node of master sequence node with ID " + masterSequenceNode.GetID() + " not found")
            return None

        return proxyNode

    def updateMaskVolume(self):
        """
        Update the mask volume based on the current mask landmarks. Returns a string that can displayed as status message.
        """
        parameterNode = self.getParameterNode()

        fanMaskMarkupsNode = self.getParameterNode().GetNodeReference(self.MASK_FAN_LANDMARKS)
        if fanMaskMarkupsNode.GetNumberOfControlPoints() < 4:
            logging.info("At least four control points are needed to define a mask")
            return("At least four control points are needed to define a mask")

        # Compute the center of the mask points

        currentVolumeNode = self.getCurrentProxyNode()
        if currentVolumeNode is None:
            logging.error("Current volume node not found")
            return("No ultrasound image is loaded")

        rasToIjk = vtk.vtkMatrix4x4()
        currentVolumeNode.GetRASToIJKMatrix(rasToIjk)

        controlPoints_ijk = np.zeros((4, 4))
        for i in range(4):
            fanMaskMarkupsNode.GetNthControlPointPosition(i, controlPoints_ijk[i, :3])
            # pad control point with 1 to make it homogeneous
            controlPoints_ijk[i, 3] = 1
            # convert to IJK
            controlPoints_ijk[i, :] = rasToIjk.MultiplyPoint(controlPoints_ijk[i, :])

        centerOfGravity = np.mean(controlPoints_ijk, axis=0)

        topLeft = np.zeros(3)
        topRight = np.zeros(3)
        bottomLeft = np.zeros(3)
        bottomRight = np.zeros(3)

        for i in range(4):
            if controlPoints_ijk[i, 0] < centerOfGravity[0] and controlPoints_ijk[i, 1] > centerOfGravity[1]:
                bottomLeft = controlPoints_ijk[i, :3]
            elif controlPoints_ijk[i, 0] > centerOfGravity[0] and controlPoints_ijk[i, 1] > centerOfGravity[1]:
                bottomRight = controlPoints_ijk[i, :3]
            elif controlPoints_ijk[i, 0] < centerOfGravity[0] and controlPoints_ijk[i, 1] < centerOfGravity[1]:
                topLeft = controlPoints_ijk[i, :3]
            elif controlPoints_ijk[i, 0] > centerOfGravity[0] and controlPoints_ijk[i, 1] < centerOfGravity[1]:
                topRight = controlPoints_ijk[i, :3]

        if np.array_equal(topLeft, np.zeros(3)) or np.array_equal(topRight, np.zeros(3)) or \
                np.array_equal(bottomLeft, np.zeros(3)) or np.array_equal(bottomRight, np.zeros(3)):
            logging.debug("Could not determine mask corners")
            return("Mask points should be in a fan or rectangular shape with two points in the top and two points in the bottom."
                   "\nMove points to try again.")

        imageArray = slicer.util.arrayFromVolume(currentVolumeNode)  # (z, y, x, channels)

        # Detect if the mask is a fan or a rectangle

        maskHeight = abs(topLeft[1] - bottomLeft[1])
        tolerancePixels = round(0.1 * maskHeight)  #todo: Make this tolerance value a setting
        if abs(topLeft[0] - bottomLeft[0]) < tolerancePixels and abs(topRight[0] - bottomRight[0]) < tolerancePixels:
            # Mask is a rectangle
            logging.info("Mask is a rectangle")
            mask_array = self.createRectangleMask(imageArray, topLeft, topRight, bottomLeft, bottomRight)
        else:
            # Mask is a fan
            logging.info("Mask is a fan")
            mask_array = self.createFanMask(imageArray, topLeft, topRight, bottomLeft, bottomRight, value=1)

        # Create a copy of the mask_array to use for computing the contour of the mask

        mask_contour_array = np.copy(mask_array)
        masking_kernel = np.ones((3, 3), np.uint8)
        mask_contour_array_eroded = cv2.erode(mask_contour_array, masking_kernel, iterations=3)
        mask_contour_array = mask_contour_array - mask_contour_array_eroded

        maskContourVolumeNode = parameterNode.GetNodeReference(self.MASK_CONTOUR_VOLUME)
        if maskContourVolumeNode is None:
            maskContourVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.MASK_CONTOUR_VOLUME)
            parameterNode.SetNodeReferenceID(self.MASK_CONTOUR_VOLUME, maskContourVolumeNode.GetID())
            maskContourVolumeNode.CreateDefaultDisplayNodes()
            maskContourDisplayNode = maskContourVolumeNode.GetDisplayNode()
            maskContourDisplayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGreen")

        maskContourVolumeNode.SetSpacing(currentVolumeNode.GetSpacing())
        maskContourVolumeNode.SetOrigin(currentVolumeNode.GetOrigin())
        ijkToRas = vtk.vtkMatrix4x4()
        currentVolumeNode.GetIJKToRASMatrix(ijkToRas)
        maskContourVolumeNode.SetIJKToRASMatrix(ijkToRas)

        slicer.util.updateVolumeFromArray(maskContourVolumeNode, mask_contour_array)

        # Add a dimension to the mask array

        mask_array = np.expand_dims(mask_array, axis=0)

        # Create a new volume node for the mask

        maskVolumeNode = parameterNode.GetNodeReference(self.MASK_VOLUME)
        if maskVolumeNode is None:
            maskVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            maskVolumeNode.SetName("Mask")
            parameterNode.SetNodeReferenceID(self.MASK_VOLUME, maskVolumeNode.GetID())

        maskVolumeNode.SetSpacing(currentVolumeNode.GetSpacing())
        maskVolumeNode.SetOrigin(currentVolumeNode.GetOrigin())
        ijkToRas = vtk.vtkMatrix4x4()
        currentVolumeNode.GetIJKToRASMatrix(ijkToRas)
        maskVolumeNode.SetIJKToRASMatrix(ijkToRas)

        slicer.util.updateVolumeFromArray(maskVolumeNode, mask_array)

        return("Mask created successfully")

    def createRectangleMask(self, imageArray, topLeft, topRight, bottomLeft, bottomRight):
        image_size_rows = imageArray.shape[1]
        image_size_cols = imageArray.shape[2]

        rectangleLeft = round((topLeft[0] + bottomLeft[0]) / 2)
        rectangleRight = round((topRight[0] + bottomRight[0]) / 2)
        rectangleTop = round((topLeft[1] + topRight[1]) / 2)
        rectangleBottom = round((bottomLeft[1] + bottomRight[1]) / 2)

        self.maskParameters = {}
        self.maskParameters["mask_type"] = "rectangle"
        self.maskParameters["rectangle_left"] = rectangleLeft
        self.maskParameters["rectangle_right"] = rectangleRight
        self.maskParameters["rectangle_top"] = rectangleTop
        self.maskParameters["rectangle_bottom"] = rectangleBottom

        mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)
        mask_array[rectangleTop:rectangleBottom, rectangleLeft:rectangleRight] = 1

        return mask_array

    def createFanMask(self, imageArray, topLeft, topRight, bottomLeft, bottomRight, value=255):
        image_size_rows = imageArray.shape[1]
        image_size_cols = imageArray.shape[2]
        mask_array = np.zeros((image_size_rows, image_size_cols), dtype=np.uint8)

        # Compute the angle of the fan mask in degrees

        if abs(topLeft[0] - bottomLeft[0]) < 0.001:
            angle1 = 90.0
        else:
            angle1 = np.arctan((topLeft[1] - bottomLeft[1]) / (topLeft[0] - bottomLeft[0])) * 180 / np.pi + 180.0
        if angle1 > 180.0:
            angle1 -= 180.0
        if angle1 < 0.0:
            angle1 += 180.0
        
        if abs(topRight[0] - bottomRight[0]) < 0.001:
            angle2 = 90.0
        else:
            angle2 = np.arctan((topRight[1] - bottomRight[1]) / (topRight[0] - bottomRight[0])) * 180 / np.pi
        if angle2 > 180.0:
            angle2 -= 180.0
        if angle2 < 0.0:
            angle2 += 180.0
        
        # Fit lines to the top and bottom points
        leftLineA, leftLineB, leftLineC = self.line_coefficients(topLeft, bottomLeft)
        rightLineA, rightLineB, rightLineC = self.line_coefficients(topRight, bottomRight)

        # Handle the case when the lines are parallel
        if leftLineB != 0 and rightLineB != 0 and leftLineA / leftLineB == rightLineA / rightLineB:
            logging.warning("Left and right lines are parallel")
            return mask_array
        
        # Compute intersection point of the two lines
        det = leftLineA * rightLineB - leftLineB * rightLineA
        if det == 0:
            logging.warning("No intersection point found")
            return mask_array

        intersectionX = (leftLineB * rightLineC - rightLineB * leftLineC) / det
        intersectionY = (rightLineA * leftLineC - leftLineA * rightLineC) / det

        # Compute average distance of top points to the intersection point

        topDistance = np.sqrt((topLeft[0] - intersectionX) ** 2 + (topLeft[1] - intersectionY) ** 2) + \
                      np.sqrt((topRight[0] - intersectionX) ** 2 + (topRight[1] - intersectionY) ** 2)
        topDistance /= 2

        # Compute average distance of bottom points to the intersection point

        bottomDistance = np.sqrt((bottomLeft[0] - intersectionX) ** 2 + (bottomLeft[1] - intersectionY) ** 2) + \
                          np.sqrt((bottomRight[0] - intersectionX) ** 2 + (bottomRight[1] - intersectionY) ** 2)
        bottomDistance /= 2

        # Mask parameters

        center_rows_px = round(intersectionY)
        center_cols_px = round(intersectionX)
        radius1 = round(topDistance)
        radius2 = round(bottomDistance)

        # Create a mask image

        # mask_array = cv2.ellipse(mask_array, (center_cols_px, center_rows_px), (radius2, radius2), 0.0, angle2, angle1, value, -1)
        mask_array = self.draw_circle_segment(mask_array, (center_cols_px, center_rows_px), radius2, angle2, angle1, value)
        mask_array = cv2.circle(mask_array, (center_cols_px, center_rows_px), radius1, 0, -1)
        
        self.maskParameters = {}
        self.maskParameters["mask_type"] = "fan"
        self.maskParameters["angle1"] = angle1
        self.maskParameters["angle2"] = angle2
        self.maskParameters["center_rows_px"] = center_rows_px
        self.maskParameters["center_cols_px"] = center_cols_px
        self.maskParameters["radius1"] = radius1
        self.maskParameters["radius2"] = radius2
        self.maskParameters["image_size_rows"] = image_size_rows
        self.maskParameters["image_size_cols"] = image_size_cols
        
        return mask_array

    def line_coefficients(self, p1, p2):
        """
        Returns the coefficients of the line equation Ax + By + C = 0
        """
        if p1[0] == p2[0]:  # Vertical line
            A = 1
            B = 0
            C = -p1[0]
        else:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            A = -m
            B = 1
            C = m * p1[0] - p1[1]
        return A, B, C

    def draw_circle_segment(self, image, center, radius, start_angle, end_angle, color):
        """
        Draws a segment of a circle with floating point start and end angles on a numpy array image.

        :param image: Image as a numpy array.
        :param center: Center of the circle (x, y).
        :param radius: Radius of the circle.
        :param start_angle: Start angle in degrees (floating point).
        :param end_angle: End angle in degrees (floating point).
        :param color: Color of the segment (B, G, R).
        :return: Image with the drawn circle segment.
        """
        mask = np.zeros_like(image)

        # Convert angles to radians
        start_angle_rad = np.deg2rad(start_angle)
        end_angle_rad = np.deg2rad(end_angle)

        # Generate points for the circle segment
        thetas = np.linspace(start_angle_rad, end_angle_rad, 360)
        xs = center[0] + radius * np.cos(thetas)
        ys = center[1] + radius * np.sin(thetas)

        # Draw the outer arc
        pts = np.array([np.round(xs), np.round(ys)]).T.astype(int)
        cv2.polylines(mask, [pts], False, color, 1)

        # Draw two lines from the center to the start and end points
        cv2.line(mask, center, tuple(pts[0]), color, 1)
        cv2.line(mask, center, tuple(pts[-1]), color, 1)

        # Fill the segment
        cv2.fillPoly(mask, [np.vstack([center, pts])], color)

        # Combine the mask with the original image
        return cv2.bitwise_or(image, mask)
    
    def maskSequence(self):
        self.updateMaskVolume()

        parameterNode = self.getParameterNode()
        currentSequenceBrowser = parameterNode.GetNodeReference(self.CURRENT_SEQUENCE)

        if currentSequenceBrowser is None:
            logging.error("No sequence browser node loaded!")
            return

        # Get mask volume

        maskVolumeNode = parameterNode.GetNodeReference(self.MASK_VOLUME)
        if maskVolumeNode is None:
            logging.error("No mask volume loaded!")
            return

        maskArray = slicer.util.arrayFromVolume(maskVolumeNode)

        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        currentSequenceBrowser.SetRecording(masterSequenceNode, True)

        for index in range(masterSequenceNode.GetNumberOfDataNodes()):
            currentSequenceBrowser.SetSelectedItemNumber(index)
            currentVolumeNode = masterSequenceNode.GetNthDataNode(index)
            currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)

            # Mask every channel of the current volume array using maskArray

            for channel in range(currentVolumeArray.shape[3]):
                currentVolumeArray[:, :, :, channel] = np.multiply(currentVolumeArray[:, :, :, channel], maskArray)

        # Re-render views

        currentSequenceBrowser.SetSelectedItemNumber(0)
        proxyNode = self.getCurrentProxyNode()
        proxyNode.GetImageData().Modified()

    def convertToJsonCompatible(self, obj):
        if isinstance(obj, MultiValue):
            return list(obj)
        if isinstance(obj, pydicom.valuerep.PersonName):
            return str(obj)
        if isinstance(obj, bytes):
            return obj.decode('latin-1')
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def compressFrameToJpeg(self, frame, quality=95):
        """
        Compress a single frame using JPEG compression.
        :param frame: a numpy array representing the image frame (W, H, C)
        :param quality: an integer from 0 to 100 setting the quality of the compression
        :return: compressed frame bytes
        """
        # If frame is 2-dimensional, expand to 3-dimensional
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)
        
        if frame.shape[2] == 1:
            image = Image.fromarray(frame[:, :, 0]).convert("L")
        else:
            image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=quality)
            return output.getvalue()

    def generateNameFromDicomData(self, ds):
        """
        Generate a filename from a DICOM header dictionary. The name will consist of two parts:
        X_Y.dcm
        X is generated by hashing the original patient UID to a 10-digit number.
        Y is generated from the DICOM instance UID, but limited to 8 digits
        
        :param headerDict: DICOM header data
        :returns: tuple (filename, patientId, instanceId)
        """
        patientUID = ds.PatientID
        instanceUID = ds.SOPInstanceUID

        if patientUID is None or patientUID == "":
            logging.error("PatientID not found in DICOM header dict")
            return ""

        if instanceUID is None or instanceUID == "":
            logging.error("SOPInstanceUID not found in DICOM header dict")
            return ""

        hash_object = hashlib.sha256()
        hash_object.update(str(patientUID).encode())
        patientId = int(hash_object.hexdigest(), 16) % 10**10

        hash_object_instance = hashlib.sha256()
        hash_object_instance.update(str(instanceUID).encode())
        instanceId = int(hash_object_instance.hexdigest(), 16) % 10**8

        # Add trailing zeros
        patientId = str(patientId).zfill(10)
        instanceId = str(instanceId).zfill(8)

        return f"{patientId}_{instanceId}.dcm", patientId, instanceId
    
    def findKeyInDict(self, d: dict, target_key: str):
        """
        Recursively search for a key in a nested dictionary.
        Returns the value if the key is found, otherwise 'N/A'.
        """
        if target_key in d:
            return d[target_key]
        
        for key, value in d.items():
            if isinstance(value, dict):
                result = self.findKeyInDict(value, target_key)
                if result != 'N/A':
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = self.findKeyInDict(item, target_key)
                        if result != 'N/A':
                            return result
        
        return 'N/A'
    
    def saveDicomFile(self, dicomFilePath, convertToGrayscale = False, compression = True):
        parameterNode = self.getParameterNode()

        # Collect all image frames in a numpy array

        currentSequenceBrowser = parameterNode.GetNodeReference(self.CURRENT_SEQUENCE)
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        proxyNode = self.getCurrentProxyNode()

        proxyNodeArray = slicer.util.arrayFromVolume(proxyNode)

        imageArray = np.zeros((masterSequenceNode.GetNumberOfDataNodes(),
                               proxyNodeArray.shape[1], proxyNodeArray.shape[2], proxyNodeArray.shape[3]), dtype=np.int8)

        for index in range(masterSequenceNode.GetNumberOfDataNodes()):
            currentSequenceBrowser.SetSelectedItemNumber(index)
            currentVolumeNode = masterSequenceNode.GetNthDataNode(index)
            currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)
            imageArray[index, :, :, :] = currentVolumeArray

        # Convert to grayscale if data is RGB and option checked
        dims = len(imageArray.shape)
        numberOfFrames = currentSequenceBrowser.GetNumberOfItems()
        if numberOfFrames > 1 and dims > 3 and convertToGrayscale:
            # Keep only R channel from RGB for grayscale conversion
            imageArray = imageArray[:,:,:,0]

        # Create a new DICOM dataset
        ds = Dataset()

        dicom_header_data = self.currentDicomHeader

        # Copy sequence of ultrasound regions from the original dicom dataset
        original_ds = self.currentDicomDataset
        if hasattr(original_ds, "SequenceOfUltrasoundRegions"):
            if len(original_ds.SequenceOfUltrasoundRegions) > 0:
                ds.SequenceOfUltrasoundRegions = original_ds.SequenceOfUltrasoundRegions

        # Copy spacing to conventional PixelSpacing tag for DICOM readers that don't support ultrasound regions
        deltaX = self.findKeyInDict(dicom_header_data, 'Physical Delta X')
        deltaY = self.findKeyInDict(dicom_header_data, 'Physical Delta Y')
        if deltaX != 'N/A' and deltaY != 'N/A':
            deltaXmm = float(deltaX) * 10
            deltaYmm = float(deltaY) * 10
            ds.PixelSpacing = [str(deltaXmm), str(deltaYmm)]

        # Verify array shape and set DICOM tags accordingly
        if len(imageArray.shape) == 4:  # Multi-frame format
            frames, height, width, channels = imageArray.shape
            ds.Rows = height
            ds.Columns = width
            ds.NumberOfFrames = frames
            ds.SamplesPerPixel = channels
        elif len(imageArray.shape) == 3:  # Muti-frame, grayscale
            frames, height, width = imageArray.shape
            ds.Rows = height
            ds.Columns = width
            ds.NumberOfFrames = frames
            ds.SamplesPerPixel = 1
        elif len(imageArray.shape) == 2:  # Single frame, Grayscale
            height, width = imageArray.shape
            ds.Rows = height
            ds.Columns = width
            ds.SamplesPerPixel = 1

        # Set PhotometricInterpretation based on number of channels
        if ds.SamplesPerPixel == 1:
            ds.PhotometricInterpretation = "MONOCHROME2"
        elif ds.SamplesPerPixel == 3:
            ds.PhotometricInterpretation = "RGB"
            if compression == True:
                ds.PhotometricInterpretation = "YBR_FULL_422"

        # Ensure the data type of numpy array matches the expected pixel data type
        ds.BitsAllocated = 8
        ds.Modality = 'US'
        
        # Set the pixel data
        if compression == True:
            # Compress each frame and collect the bytes
            compressed_frames = []
            for frame in imageArray:
                compressed_frame = self.compressFrameToJpeg(frame)
                compressed_frames.append(compressed_frame)
            ds.PixelData = encapsulate(compressed_frames)
            ds['PixelData'].is_undefined_length = True
            ds.LossyImageCompression = '01'
            ds.LossyImageCompressionMethod = 'ISO_10918_1'
        else:
            ds.PixelData = imageArray.tobytes()
        
        if hasattr(original_ds, "Manufacturer") and original_ds.Manufacturer:
            ds.Manufacturer = original_ds.Manufacturer
        
        # Find and set the following DICOM tags from the JSON header file:
        for key in dicom_header_data.keys():
            if "patient" in key.lower() and "name" in key.lower():
                ds.PatientName = str(dicom_header_data[key])
            if "patient id" in key.lower():
                ds.PatientID = str(dicom_header_data[key])
            if "study id" in key.lower():
                ds.StudyID = str(dicom_header_data[key])
            if "study date" in key.lower():
                ds.StudyDate = dicom_header_data[key]
            if "study time" in key.lower():
                ds.StudyTime = dicom_header_data[key]
            if "study description" in key.lower():
                ds.StudyDescription = dicom_header_data[key]
            if "patient" in key.lower() and "sex" in key.lower():
                ds.PatientSex = dicom_header_data[key]
            if "patient" in key.lower() and "age" in key.lower():
                ds.PatientAge = dicom_header_data[key]
            if "series number" in key.lower():
                ds.SeriesNumber = dicom_header_data[key]
            if "study instance uid" in key.lower():
                ds.StudyInstanceUID = dicom_header_data[key]
            # If we keep the SeriesInstanceUID from the original DICOM header, only one series is displayed in the DICOM viewer
            # Either comment out the next two lines, or patch the original DICOM to generate different series IDs for different instances.
            if "series instance uid" in key.lower():
                ds.SeriesInstanceUID = dicom_header_data[key]
            else:
                ds.SeriesInstanceUID = generate_uid()
            if "sop Instance uid" in key.lower():
                ds.SOPInstanceUID = dicom_header_data[key]
            else:
                ds.SOPInstanceUID = generate_uid()
            if "sop class uid" in key.lower():
                ds.SOPClassUID = dicom_header_data[key]
            if "manufacturer" in key.lower() and "model name" in key.lower():
                ds.ManufacturerModelName = dicom_header_data[key]
            if "transducer" in key.lower():
                ds.TransducerType = dicom_header_data[key]
            if "station name" in key.lower():
                ds.StationName = dicom_header_data[key]
        
        # Set default SOPClassUID if not present
        if not hasattr(ds, 'SOPClassUID'):
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # UID for Secondary Capture Image
            logging.warning(f"SOPClassUID not found. Using default UID {ds.SOPClassUID}.")

        # Set meta information for the DICOM file
        meta = Dataset()
        meta.MediaStorageSOPClassUID = ds.SOPClassUID  # This should match the SOPClassUID of the dataset
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID  # This should match the SOPInstanceUID of the dataset
        meta.ImplementationClassUID = generate_uid(None)  # Generate a new UID for our implementation
        if compression == True:
            meta.TransferSyntaxUID = JPEGBaseline
        else:
            meta.TransferSyntaxUID = ImplicitVRLittleEndian  # This is the default transfer syntax

        file_ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)

        # Copy over the dataset values from ds to file_ds
        for elem in ds:
            file_ds.add(elem)

        # Set the is_implicit_VR and is_little_endian attributes for the encoding
        if compression == True:
            file_ds.is_implicit_VR = False
            file_ds.is_little_endian = True
        else:
            file_ds.is_implicit_VR = True
            file_ds.is_little_endian = True

        # Save the DICOM file
        file_ds.save_as(dicomFilePath)
        logging.info(f"DICOM generated successfully: {dicomFilePath}")
        
    def exportDicom(self,
                    outputDirectory,
                    outputFilename = None,
                    convertToGrayscale = False,
                    labels = None,
                    compression = True):
        """
        Export image array to DICOM files.

        :param outputDirectory: Output directory where the DICOM files will be saved.
        :param outputFilename: Output file name without extension. If None, a file name will be generated based
            on patient ID and instance UID.
        :param convertToGrayscale: If True, convert RGB images to grayscale.
        :param labels: List of annotation labels to be saved in accompanying CSV file.
        :param compression: If True, use JPEG compression (minimal) in output DICOM.
        """                
        # Record sequence information to a dictionary. This will be saved in the annotations JSON file.
        SOPInstanceUID = self.currentDicomDataset.SOPInstanceUID
        if SOPInstanceUID is None:
            SOPInstanceUID = "None"
        sequenceInfo = {
            'SOPInstanceUID': SOPInstanceUID,
            'GrayscaleConversion': False
        }

        if convertToGrayscale:
            sequenceInfo['GrayscaleConversion'] = True

        # Create output directory if it doesn't exist
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)

        # Save DICOM image file
        if outputFilename is None:
            outputFilename, _, _ = self.generateNameFromDicomData(self.currentDicomDataset)
        if outputFilename == "":
            return None, None, None
        dicomFilePath = os.path.join(outputDirectory, outputFilename)
        self.saveDicomFile(dicomFilePath, convertToGrayscale=convertToGrayscale, compression=compression)
        
        # Save original DICOM header to a json file. This may not be completely anonymized.
        dicomHeaderFileName = outputFilename.replace(".dcm", "_DICOMHeader.json")
        dicomHeaderFilePath = os.path.join(outputDirectory, dicomHeaderFileName)
        with open(dicomHeaderFilePath, 'w') as outfile:
            json.dump(self.currentDicomHeader, outfile, default=self.convertToJsonCompatible)
        
        # Add mask parameters to sequenceInfo
        for key, value in self.maskParameters.items():
            sequenceInfo[key] = value
        
        # Add annotation labels to sequenceInfo
        if labels is not None:
            sequenceInfo["AnnotationLabels"] = labels

        # Save sequenceInfo to a file
        annotationsFilename = outputFilename.replace(".dcm", ".json")
        sequenceInfoFilePath = os.path.join(outputDirectory, annotationsFilename)
        with open(sequenceInfoFilePath, 'w') as outfile:
            json.dump(sequenceInfo, outfile)

        return dicomFilePath, sequenceInfoFilePath, dicomHeaderFilePath

    def exportArrays(self, outputDirectory = None, outputFilename = None, convertToGrayscale = False, labels = None, compression = False):
        """
        DEPRECATED: The module only exports in DICOM format. This function may be removed in the future.

        Export image array to a file in numpy format.

        :param outputDirectory: Output directory where the image array file will be saved.
        :param outputFilename: Output file name without extension. If None, a file name will be generated based on patient ID and sequence name.
        :param convertToGrayscale: If True, convert RGB images to grayscale.
        :param labels: List of annotation labels.
        :param compression: If True, save the image array as a compressed numpy array.
        :return: Tuple of file paths: (imageArrayFilePath, sequenceInfoFilePath, dicomHeaderFilePath)        
        """
        parameterNode = self.getParameterNode()

        # Collect all image frames in a numpy array

        currentSequenceBrowser = parameterNode.GetNodeReference(self.CURRENT_SEQUENCE)
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        proxyNode = self.getCurrentProxyNode()

        proxyNodeArray = slicer.util.arrayFromVolume(proxyNode)

        imageArray = np.zeros((masterSequenceNode.GetNumberOfDataNodes(),
                               proxyNodeArray.shape[1], proxyNodeArray.shape[2], proxyNodeArray.shape[3]), dtype=np.int8)

        for index in range(masterSequenceNode.GetNumberOfDataNodes()):
            currentSequenceBrowser.SetSelectedItemNumber(index)
            currentVolumeNode = masterSequenceNode.GetNthDataNode(index)
            currentVolumeArray = slicer.util.arrayFromVolume(currentVolumeNode)
            imageArray[index, :, :, :] = currentVolumeArray

        # If output directory is not given, try to get it from application settings

        if outputDirectory is None:
            settings = qt.QSettings()
            outputDirectory = settings.value('AnonymizeUltrasound/OutputDirectory', '')
            if outputDirectory == '':
                outputDirectory = qt.QDir.homePath()

        # Create output directory if it doesn't exist

        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)

        # Generate a file name for the image array file using patient ID and sequence name
        currentSequenceBrowserName = currentSequenceBrowser.GetName()

        # Remove characters that are not allowed in file names
        currentSequenceBrowserName = currentSequenceBrowserName.replace(" ", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace(":", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace("/", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace("\\", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace("*", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace("?", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace("\"", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace("<", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace(">", "_")
        currentSequenceBrowserName = currentSequenceBrowserName.replace("|", "_")

        if outputFilename is None:
            imageArrayFileName = f"{parameterNode.GetParameter(self.CURRENT_PATIENT_ID)}_{currentSequenceBrowserName}.npy"
        else:
            imageArrayFileName = outputFilename + ".npy"

        imageArrayFilePath = os.path.join(outputDirectory, imageArrayFileName)
        
        # Record sequence related information to a dictionary
        sequenceInfo = {
            'PatientID': parameterNode.GetParameter(self.CURRENT_PATIENT_ID),
            'SequenceName': currentSequenceBrowser.GetName(),
            'GrayscaleConversion': False
        }

        # Convert to grayscale if data is RGB and option checked
        dims = len(imageArray.shape)
        numberOfFrames = currentSequenceBrowser.GetNumberOfItems()
        if numberOfFrames > 1 and dims > 3 and convertToGrayscale:
            # Keep only R channel from RGB for grayscale conversion
            imageArray = imageArray[:,:,:,0]
            sequenceInfo['GrayscaleConversion'] = True

        # Save image array to file.
        # If compression setting is enabled, then save as compressed numpy array.
        if compression:
            # Change the extension to .npz
            imageArrayFilePath = os.path.splitext(imageArrayFilePath)[0] + ".npz"
            np.savez_compressed(imageArrayFilePath, imageArray)
        else:
            np.save(imageArrayFilePath, imageArray)

        # Add mask parameters to sequenceInfo
        for key, value in self.maskParameters.items():
            sequenceInfo[key] = value
        if outputFilename is None:
            sequenceInfoFileName = f"{parameterNode.GetParameter(self.CURRENT_PATIENT_ID)}_{currentSequenceBrowserName}.json"
        else:
            sequenceInfoFileName = outputFilename + ".json"
        sequenceInfoFilePath = os.path.join(outputDirectory, sequenceInfoFileName)

        # Add annotation labels to sequenceInfo
        if labels is not None:
            sequenceInfo["AnnotationLabels"] = labels

        # Save sequenceInfo to a file
        with open(sequenceInfoFilePath, 'w') as outfile:
            json.dump(sequenceInfo, outfile)

        # Save current DICOM header dictionary to a file
        if outputFilename is None:
            dicomHeaderFileName = f"{parameterNode.GetParameter(self.CURRENT_PATIENT_ID)}_{currentSequenceBrowserName}_DICOMHeader.json"
        else:
            dicomHeaderFileName = outputFilename + "_DICOMHeader.json"
        dicomHeaderFilePath = os.path.join(outputDirectory, dicomHeaderFileName)

        with open(dicomHeaderFilePath, 'w') as outfile:
            json.dump(self.currentDicomHeader, outfile, default=self.convertToJsonCompatible)

        return imageArrayFilePath, sequenceInfoFilePath, dicomHeaderFilePath

    def importData(self, inputDirectory):
        """
        Import folder in temporary DICOM database and return loaded node IDs.
        """
        # instantiate a new DICOM browser
        slicer.util.selectModule("DICOM")
        dicomBrowser = slicer.modules.DICOMWidget.browserWidget.dicomBrowser

        dicomBrowser.importDirectory(inputDirectory, dicomBrowser.ImportDirectoryAddLink)

        dicomBrowser.waitForImportFinished()

        db = slicer.dicomDatabase
        patientUIDs = db.patients()

        return patientUIDs

    def needToLoadPatient(self):
        logging.warning("needToLoadPatient is deprecated, use loadNextSequence instead")
        return False
    
        # parameterNode = self.getParameterNode()
        # currentSequenceBrowser = parameterNode.GetNodeReference(self.CURRENT_SEQUENCE)
        # if not currentSequenceBrowser:
        #     return True
        # currentSequenceIndex = self.loadedSequenceBrowserNodeIds.index(currentSequenceBrowser.GetID())
        # return currentSequenceIndex == len(self.loadedSequenceBrowserNodeIds) - 1

    def filesForSeries(seriesUID):
        """
        Returns a list of file paths for the given series UID.
        """
        db = slicer.dicomDatabase
        files = []
        for instanceUID in db.fileLists(seriesUID):
            files.append(db.fileForInstance(instanceUID))
        return files

    def resetScene(self, leaveMaskFiducials=True):
        """
        Reset the scene by clearing it and setting it up again.

        leaveMaskFiducials: If True, then leave the mask fiducials in the scene. If False, then remove them.
        """
        parameterNode = self.getParameterNode()
        maskFiducialsList = []

        if leaveMaskFiducials:
            # Save mask fiducials    
            maskFiducials = parameterNode.GetNodeReference(self.MASK_FAN_LANDMARKS)
            if maskFiducials is not None:
                # Store all control point coordinates in a list
                for i in range(maskFiducials.GetNumberOfControlPoints()):
                    coord = [0, 0, 0]
                    maskFiducials.GetNthControlPointPosition(i, coord)
                    maskFiducialsList.append(coord)
        
        logging.info(f"Saved {len(maskFiducialsList)} mask fiducials")

        # Clear the scene
        slicer.mrmlScene.Clear(0)
        self.currentDicomDataset = None
        self.currentDicomHeader = None
        self.setupScene(maskControlPointsList=maskFiducialsList)

        logging.info(f"Restored {len(maskFiducialsList)} mask fiducials")   
        
    def loadNextSequence(self):
        """
        Load next sequence in the list of DICOM files.
        Returns the index of the loaded sequence in the dataframe of DICOM files, or None if no more sequences are available.
        """
        self.resetScene()
        
        parameterNode = self.getParameterNode()

        # Get next filepath from dicomDf. If nextDicomDfIndex is larger than the number of rows in dicomDf, then
        # return None.
        if self.nextDicomDfIndex >= len(self.dicomDf):
            return None
        nextDicomDfRow = self.dicomDf.iloc[self.nextDicomDfIndex]

        # Make sure a temporary folder for the DICOM files exists
        tempDicomDir = slicer.app.temporaryPath + '/AnonymizeUltrasound'
        logging.info("Temporary DICOM directory: " + tempDicomDir)
        if not os.path.exists(tempDicomDir):
            os.makedirs(tempDicomDir)
        
        # Delete all files in the temporary folder
        for file in os.listdir(tempDicomDir):
            os.remove(os.path.join(tempDicomDir, file))
        
        # Copy DICOM file to temporary folder
        shutil.copy(nextDicomDfRow['Filepath'], tempDicomDir)
        
        # Patch the DICOM file to add spacing information if it is missing, but available from other rows
        temporaryDicomFilepath = os.path.join(tempDicomDir, os.path.basename(nextDicomDfRow['Filepath']))
        to_patch = nextDicomDfRow['Patch']
        if to_patch:
            physical_delta_x = nextDicomDfRow['PhysicalDeltaX']
            physical_delta_y = nextDicomDfRow['PhysicalDeltaY']
            if physical_delta_x is not None and physical_delta_y is not None:
                ds = pydicom.dcmread(temporaryDicomFilepath)
                ds.PixelSpacing = [str(physical_delta_x*10.0), str(physical_delta_y*10.0)]  # Convert from cm/pixel to mm/pixel
                ds.save_as(temporaryDicomFilepath)
                logging.info(f"Patched DICOM file {temporaryDicomFilepath} with physical delta X and Y")

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
                parameterNode.SetNodeReferenceID(self.CURRENT_SEQUENCE, currentSequenceBrowser.GetID())
                self.currentDicomHeader = self.dicomHeaderDictForBrowserNode(currentSequenceBrowser)
                # Add a "DicomFile" attribute to the currentSequenceBrowser, so we can check the source file later for information
                currentSequenceBrowser.SetAttribute("DicomFile", nextDicomDfRow['Filepath'])
                if self.currentDicomHeader is None:
                    logging.error(f"Could not find DICOM header for sequence browser node {currentSequenceBrowser.GetID()}")
                break

        # If accidentally pixel data gets in the dicom header, remove it
        if "Pixel Data" in self.currentDicomHeader:
            del self.currentDicomHeader["Pixel Data"]

        # Increment nextDicomDfIndex
        self.nextDicomDfIndex += 1

        # Delete files from temporary folder
        for file in os.listdir(tempDicomDir):
            os.remove(os.path.join(tempDicomDir, file))
        
        # Make this sequence browser node the current one in the toolbar
        slicer.modules.sequences.setToolBarActiveBrowserNode(currentSequenceBrowser)

        # Get the proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = currentSequenceBrowser.GetMasterSequenceNode()
        if masterSequenceNode is None:
            logging.error("Master sequence node of sequence browser node with ID " + currentSequenceBrowser.GetID() + " not found")
            return None
        proxyNode = currentSequenceBrowser.GetProxyNode(masterSequenceNode)
        if proxyNode is None:
            logging.error("Proxy node of master sequence node with ID " + masterSequenceNode.GetID() + " not found")
            return None

        # If proxyNode is a vtkMRMLScalarVolumeNode or vtkMRMLVectorVolumeNode, then set it as the background volume
        if proxyNode.IsA("vtkMRMLScalarVolumeNode") or proxyNode.IsA("vtkMRMLVectorVolumeNode"):
            backgroundVolumeNode = proxyNode
            layoutManager = slicer.app.layoutManager()
            sliceLogic = layoutManager.sliceWidget('Red').sliceLogic()
            compositeNode = sliceLogic.GetSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID(backgroundVolumeNode.GetID())
        
        return self.nextDicomDfIndex - 1

    def dicomHeaderToDict(self, ds, parent=None):
        """
        Convert a DICOM dataset to a Python dictionary.
        """
        if parent is None:
            parent = {}
        for elem in ds:
            if elem.VR == "SQ":
                parent[elem.name] = []
                for item in elem:
                    child = {}
                    self.dicomHeaderToDict(item, child)
                    parent[elem.name].append(child)
            else:
                parent[elem.name] = elem.value
        return parent

    def anonymizeDicomHeaderDict(self, dicomHeaderDict):
        """
        Anonymize DICOM header dictionary.
        """
        keysToAnonymize = {
            "Accession Number": "AccessionNumber",
            "Other Patient IDs": "OtherPatientIDs",
            "Patient's Address": "PatientAddress",
            "Patient's Name": "PatientName",
            "Patient ID": "PatientID",
            "Patient's Birth Date": "00000000",
            "Referring Physician's Name": "ReferringPhysicianName",
            "Study Date": "StudyDate",
            "Study Time": "StudyTime",
            "Series Date": "SeriesDate",
            "Series Time": "SeriesTime",
            "Series Number": "SeriesNumber",
        }

        # Loop through all keys in the dictionary and replace the ones that need to be anonymized
        # with the corresponding key from the keysToAnonymize dictionary.
        # If the value is a dictionary, then call this function recursively.
        # If the value is a list, then call this function for each item in the list.

        anonymizedDicomHeaderDict = {}
        for key, value in dicomHeaderDict.items():
            if key in keysToAnonymize:
                anonymizedDicomHeaderDict[key] = keysToAnonymize[key]
            elif isinstance(value, dict):
                self.anonymizeDicomHeaderDict(value)
            elif isinstance(value, list):
                for item in value:
                    self.anonymizeDicomHeaderDict(item)
            else:
                anonymizedDicomHeaderDict[key] = value
        return anonymizedDicomHeaderDict

    def dicomHeaderDictForBrowserNode(self, browserNode):
        """
        Return DICOM header for the given browser node.
        Also sets up self.currentDicomDataset variable as a pydicom dataset with same contents as returned dictionary.

        :param browserNode: Sequence browser node
        :return: DICOM header dictionary
        """
        if browserNode is None:
            return None

        # Get the proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = browserNode.GetMasterSequenceNode()
        proxyNode = browserNode.GetProxyNode(masterSequenceNode)

        # Get DICOM.instanceUID attribute from proxy node
        instanceUID = proxyNode.GetAttribute("DICOM.instanceUIDs")
        if instanceUID is None:
            logging.error("DICOM.instanceUIDs attribute not found in proxy node")
            return None

        # If instanceUID is a list, keep only the first item
        if isinstance(instanceUID, list):
            instanceUID = instanceUID[0]

        # Find row in self.dicomDf where instanceUID matches first instanceUID in instanceUIDs
        filepath = self.getFileForBrowserNode(browserNode)
        if filepath is None:
            logging.error(f"Could not find DICOM file for instanceUID {instanceUID}")
            return None

        ds = pydicom.dcmread(filepath)
        dsInstanceUID = ds["0008", "0018"].value
        if dsInstanceUID == instanceUID:
            self.currentDicomDataset = pydicom.dcmread(filepath)
            self.anonymizeDicomDataset(ds)
            dicomHeaderDict = self.dicomHeaderToDict(ds)
            return dicomHeaderDict

        return None

    def getFileForBrowserNode(self, browserNode):
        if browserNode is None:
            return None

        # Get the proxy node of the master sequence node of the selected sequence browser node
        masterSequenceNode = browserNode.GetMasterSequenceNode()
        proxyNode = browserNode.GetProxyNode(masterSequenceNode)

        # Get DICOM.instanceUID attribute from proxy node
        instanceUID = proxyNode.GetAttribute("DICOM.instanceUIDs")
        if instanceUID is None:
            logging.error("DICOM.instanceUIDs attribute not found in proxy node")
            return None

        # If instanceUID is a list, keep only the first item
        if isinstance(instanceUID, list):
            instanceUID = instanceUID[0]

        # Find row in self.dicomDf where instanceUID matches first instanceUID in instanceUIDs
        filepath = None
        for index, row in self.dicomDf.iterrows():
            rowInstanceUID = row['InstanceUID']
            currentInstanceUID = instanceUID
            if rowInstanceUID == currentInstanceUID:
                filepath = row['Filepath']
                break
        if filepath is None:
            logging.error(f"Could not find DICOM file for instanceUID {instanceUID}")
            return None

        return filepath

    def currentRelativePath(self, inputDicomFolder):
        """
        Return path of DICOM file of current sequence. The path is relative to the input dicom folder
        """
        parameterNode = self.getParameterNode()
        currentBrowserNode = parameterNode.GetNodeReference(self.CURRENT_SEQUENCE)
        dicomFile = self.getFileForBrowserNode(currentBrowserNode)
        if dicomFile is None:
            logging.error(f"Could not find DICOM file for current sequence!")
            return None

        relativePath = os.path.relpath(dicomFile, inputDicomFolder)

        # Save file name without extension
        filenameWithoutExtension = os.path.splitext(os.path.basename(relativePath))[0]

        # Remove filename from relative path
        relativePath = os.path.dirname(relativePath)

        return relativePath, filenameWithoutExtension

    def person_names_callback(self, dataset, data_element):
        if data_element.VR == "PN":
            data_element.value = "anonymous"

    def anonymizeDicomDataset(self, ds, removePrivateTags = False):
        """Anonymize a DICOM dataset"""

        _, patientId, instanceId = self.generateNameFromDicomData(ds)

        # Remove patient name and any other person names
        ds.walk(self.person_names_callback)
        ds.PatientID = patientId
        ds.PatientName = patientId

        # Remove data elements (should only do so if DICOM type 3 optional)
        # Use general loop so easy to add more later
        # Could also have done: del ds.OtherPatientIDs, etc.
        for name in ["OtherPatientIDs", "OtherPatientIDsSequence"]:
            if name in ds:
                delattr(ds, name)

        # Keep only year from patient birth date

        if "PatientBirthDate" in ds:
            ds.PatientBirthDate = ds.PatientBirthDate[0:4] + "0101"

        # Remove private tags
        if removePrivateTags:
            ds.remove_private_tags()
        


#
# AnonymizeUltrasoundTest
#

class AnonymizeUltrasoundTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_AnonymizeUltrasound1()

    def test_AnonymizeUltrasound1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
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

        # Get/create input data ...


        self.delayDisplay('Test passed')
