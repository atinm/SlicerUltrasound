# Python testing configuration for SlicerUltrasound extension
# This file configures integration and DICOM loading tests

if(BUILD_TESTING)
  # DICOM loading integration test
  slicer_add_python_unittest(SCRIPT test_dicom_loading.py)
  set_tests_properties(test_dicom_loading PROPERTIES
    LABELS "integration;dicom;ultrasound"
    TIMEOUT 300
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/Slicer-4.11/qt-scripted-modules"
  )

  # Debug script (not a test, but useful for development)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/debug_dicom_import.py
    ${CMAKE_BINARY_DIR}/bin/debug_dicom_import.py
    COPYONLY
  )

  # Synthetic DICOM creator (utility script)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/create_synthetic_dicom.py
    ${CMAKE_BINARY_DIR}/bin/create_synthetic_dicom.py
    COPYONLY
  )

  # Test data directory
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_data
    ${CMAKE_BINARY_DIR}/bin/test_data
    COPYONLY
  )
endif()