# Testing Infrastructure

This directory contains the testing infrastructure for the SlicerUltrasound extension.

## Directory Structure

```
Testing/
├── Python/
│   ├── test_dicom_loading.py      # DICOM loading integration tests
│   ├── debug_dicom_import.py      # Interactive DICOM debugging
│   ├── create_synthetic_dicom.py  # Test data generation utility
│   ├── test_data/                 # Sample DICOM files and annotations
│   └── CMakeLists.txt            # Python test configuration
├── CMakeLists.txt                # Main testing configuration
└── README.md                     # This file
```

## Test Categories

### Integration Tests (`Testing/Python/`)

- **DICOM Loading Tests** - Test DICOM import, annotation loading, and sequence navigation
- **Debug Scripts** - Interactive debugging tools for development
- **Test Data** - Sample DICOM files and annotations for reproducible testing

### Module Tests (`ModuleName/Testing/Python/`)

- **Unit Tests** - Test individual module logic and functions
- **GUI Tests** - Test user interface interactions
- **Module-specific Tests** - Tests specific to each module's functionality

## Running Tests

### Using Makefile (Recommended)

```bash
# Run all tests
make test

# Run specific test types
make test-dicom          # DICOM loading tests
make test-gui            # GUI interaction tests
make test-slicer         # Slicer-native tests

# Debug and development
make debug-dicom         # Interactive DICOM debugging
make check-env           # Environment validation
```

### Using CTest

```bash
# Build with testing
make build-testing

# Run all tests
cd build && ctest -V

# Run specific test categories
cd build && ctest -L integration
cd build && ctest -L unit
cd build && ctest -L gui
```

## Test Data

The `test_data/` directory contains:

- Sample DICOM files for testing
- Annotation files in JSON format
- Multiple rater annotations for testing adjudication

## Development

### Adding New Tests

1. **Integration Tests**: Add to `Testing/Python/`
2. **Module Tests**: Add to `ModuleName/Testing/Python/`
3. **Update CMakeLists.txt**: Configure test properties and labels
4. **Update Makefile**: Add convenient targets if needed

### Test Best Practices

- Use descriptive test names
- Test both success and failure cases
- Include edge case testing
- Keep tests independent and isolated
- Clean up resources properly
- Use appropriate test labels for categorization

## Environment Requirements

- Slicer 5.0+ with Python support
- Display for GUI tests
- Test data files in `test_data/` directory

## Notes

- All tests run in Slicer's embedded Python environment
- Tests properly clean up and quit Slicer after completion
- Test data is included for reproducible testing
- Debug scripts are available for development troubleshooting
