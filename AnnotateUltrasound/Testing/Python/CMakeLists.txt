slicer_add_python_unittest(
  NAME AnnotateUltrasoundLogicTest
  SCRIPT AnnotateUltrasoundLogicTest.py
  SLICER_ARGS --no-splash --no-main-window
  PROPERTIES
    LABELS "unit;slicer;ultrasound"
    TIMEOUT 300
)

slicer_add_python_unittest(
  NAME AnnotateUltrasoundWidgetTest
  SCRIPT AnnotateUltrasoundWidgetTest.py
  SLICER_ARGS --no-splash --no-main-window
  PROPERTIES
    LABELS "integration;gui;slicer;ultrasound"
    TIMEOUT 600
    ENVIRONMENT "DISPLAY=:0"
)
