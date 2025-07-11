slicer_add_python_unittest(
  NAME AnonymizeUltrasoundModuleTest
  SCRIPT AnonymizeUltrasoundModuleTest.py
  SLICER_ARGS --no-splash --no-main-window
  PROPERTIES
    LABELS "unit;slicer;ultrasound"
    TIMEOUT 300
)
