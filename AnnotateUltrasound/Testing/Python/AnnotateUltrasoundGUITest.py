#!/usr/bin/env python3
"""
GUI test for AnnotateUltrasound module using Slicer's GUI test harness.
This test simulates user interactions and verifies UI behavior.
"""

import sys
import os
import slicer
import vtk
import time

# Add the module path to sys.path
modulePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, modulePath)

# Import the module directly
from AnnotateUltrasound import AnnotateUltrasoundLogic, AnnotateUltrasoundWidget


class AnnotateUltrasoundGUITest:
    """
    GUI test for AnnotateUltrasound module.
    This test simulates user interactions and verifies UI behavior.
    """

    def __init__(self):
        self.widget = None
        self.logic = None

    def setUp(self):
        """Set up the test environment."""
        slicer.mrmlScene.Clear(0)

        # Create the widget directly (not through slicer.modules)
        self.widget = AnnotateUltrasoundWidget()
        self.logic = self.widget.logic

        # Wait for UI to be ready
        time.sleep(1)

    def tearDown(self):
        """Clean up after tests."""
        if self.widget:
            del self.widget
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run all GUI tests."""
        print("=== Running AnnotateUltrasound GUI Tests ===")

        self.setUp()

        try:
            self.test_minimal_rater_persistence()
            self.test_widget_creation()
            self.test_button_interactions()
            self.test_rater_table_interactions()
            self.test_keyboard_shortcuts()
            self.test_line_creation_workflow()

            print("✅ All GUI tests passed!")

        except Exception as e:
            print(f"❌ GUI test failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.tearDown()

    def test_minimal_rater_persistence(self):
        """Minimal test: set rater name, press Enter, click addPleuraButton, print state before and after."""
        print("Testing minimal rater persistence...")

        # Ensure parameter node is set
        if self.widget._parameterNode is None and self.widget.logic:
            self.widget.setParameterNode(self.widget.logic.getParameterNode())

        self.widget.ui.raterName.setText("minimal_rater")
        print(f"After setText rater name: '{self.widget.ui.raterName.text}'")

        self.widget.ui.raterName.returnPressed.emit()
        print(f"After returnPressed parameter node rater: '{self.widget._parameterNode.rater if self.widget._parameterNode else 'None'}'")
        time.sleep(0.5)
        print(f"Before addPleuraButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater if self.widget._parameterNode else 'None'}'")
        self.widget.ui.addPleuraButton.click()
        time.sleep(0.1)
        print(f"After addPleuraButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater if self.widget._parameterNode else 'None'}'")
        print("✅ Minimal rater persistence test complete")

    def test_widget_creation(self):
        """Test that the widget can be created and initialized."""
        print("Testing widget creation...")

        # Verify widget was created
        self.assertIsNotNone(self.widget)
        self.assertIsNotNone(self.logic)

        # Debug: Print UI attributes to see what's available
        print("UI object:", self.widget.ui)
        print("UI type:", type(self.widget.ui))
        print("UI attributes:", dir(self.widget.ui))

        # Verify UI elements exist
        self.assertIsNotNone(self.widget.ui)
        self.assertIsNotNone(self.widget.ui.addPleuraButton)
        self.assertIsNotNone(self.widget.ui.addBlineButton)
        self.assertIsNotNone(self.widget.ui.raterColorTable)

        print("✅ Widget creation test passed")

    def test_button_interactions(self):
        """Test button clicks and state changes."""
        print("Testing button interactions...")

        # Debug: Check initial state
        print(f"Initial rater name: '{self.widget.ui.raterName.text}'")
        print(f"Initial parameter node rater: '{self.widget._parameterNode.rater}'")

        # Set a rater name before annotation actions
        self.widget.ui.raterName.setText("test_rater")
        print(f"After setText rater name: '{self.widget.ui.raterName.text}'")

        # Simulate pressing Enter to trigger returnPressed signal
        self.widget.ui.raterName.returnPressed.emit()
        print(f"After returnPressed parameter node rater: '{self.widget._parameterNode.rater}'")

        # Wait for logic to update
        time.sleep(0.5)  # Increased delay

        # Test Add Pleura button
        self.widget.onRaterNameChanged()  # Ensure rater is synced before clicking
        print(f"Before addPleuraButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater}'")
        initial_state = self.widget.ui.addPleuraButton.isChecked()
        self.widget.ui.addPleuraButton.click()
        time.sleep(0.1)  # Allow UI to update

        # Button should change state
        new_state = self.widget.ui.addPleuraButton.isChecked()
        self.assertNotEqual(initial_state, new_state)

        # Test Add B-line button
        self.widget.onRaterNameChanged()  # Ensure rater is synced before clicking
        print(f"Before addBlineButton click: UI text='{self.widget.ui.raterName.text}', parameter node rater='{self.widget._parameterNode.rater}'")
        initial_state = self.widget.ui.addBlineButton.isChecked()
        self.widget.ui.addBlineButton.click()
        time.sleep(0.1)

        new_state = self.widget.ui.addBlineButton.isChecked()
        self.assertNotEqual(initial_state, new_state)

        print("✅ Button interactions test passed")

    def test_rater_table_interactions(self):
        """Test rater table interactions."""
        print("Testing rater table interactions...")

        # Verify table exists
        table = self.widget.ui.raterColorTable
        self.assertIsNotNone(table)

        # Test adding a rater (if there's an add rater button)
        if hasattr(self.widget.ui, 'addRaterButton'):
            initial_row_count = table.rowCount()
            self.widget.ui.addRaterButton.click()
            time.sleep(0.1)

            new_row_count = table.rowCount()
            self.assertGreater(new_row_count, initial_row_count)

        print("✅ Rater table interactions test passed")

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts."""
        print("Testing keyboard shortcuts...")

        # Verify shortcuts exist
        self.assertIsNotNone(self.widget.shortcutW)
        self.assertIsNotNone(self.widget.shortcutS)
        self.assertIsNotNone(self.widget.shortcutSpace)
        self.assertIsNotNone(self.widget.shortcutE)
        self.assertIsNotNone(self.widget.shortcutD)
        self.assertIsNotNone(self.widget.shortcutA)

        # Note: Simulating key events requires PySide2 which may not be available
        # in all Slicer environments. For now, we just verify the shortcuts exist.
        print("✅ Keyboard shortcuts test passed")

    def test_line_creation_workflow(self):
        """Test the complete line creation workflow."""
        print("Testing line creation workflow...")

        # Set up test data
        test_raters = {"rater1", "rater2"}
        self.logic.setSelectedRaters(test_raters)

        # Verify raters are set
        selected = self.logic.getSelectedRaters()
        self.assertEqual(selected, test_raters)

        # Test creating a pleura line
        coordinates = [[0, 0, 0], [1, 1, 1]]
        line = self.logic.createMarkupLine("test_pleura", "test_rater", coordinates)
        self.assertIsNotNone(line)

        # Test creating a B-line
        line = self.logic.createMarkupLine("test_bline", "test_rater", coordinates)
        self.assertIsNotNone(line)

        print("✅ Line creation workflow test passed")

    # Helper assertion methods
    def assertIsNotNone(self, obj):
        if obj is None:
            raise AssertionError(f"Expected not None, but got None")

    def assertIsNone(self, obj):
        if obj is not None:
            raise AssertionError(f"Expected None, but got {obj}")

    def assertEqual(self, a, b):
        if a != b:
            raise AssertionError(f"Expected {a}, but got {b}")

    def assertNotEqual(self, a, b):
        if a == b:
            raise AssertionError(f"Expected not {a}, but got {b}")

    def assertGreater(self, a, b):
        if not a > b:
            raise AssertionError(f"Expected {a} > {b}")

    def assertGreaterEqual(self, a, b):
        if not a >= b:
            raise AssertionError(f"Expected {a} >= {b}")


def runGUITest():
    """Run the GUI test."""
    test = AnnotateUltrasoundGUITest()
    test.runTest()


if __name__ == "__main__":
    runGUITest()
