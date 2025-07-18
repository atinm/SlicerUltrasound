# Tests Directory

This directory contains comprehensive unit tests for the SlicerUltrasound project. The tests are organized by module and designed to run quickly without requiring Slicer or external dependencies.

## Test Organization

### Module-Specific Tests

#### `AnnotateUltrasound/tests/` (45 tests)

- **`test_annotate_ultrasound_logic.py`** (9 tests) - Core annotation logic
- **`test_cache_management.py`** (10 tests) - Cache management for markup data
- **`test_reload_functionality.py`** (11 tests) - Data reloading and state preservation
- **`test_overlay_management.py`** (13 tests) - Overlay rendering and visualization
- **`test_dicom_loading.py`** (2 tests) - DICOM loading functionality (in Testing/Python/)

#### `AnonymizeUltrasound/tests/` (13 tests)

- **`test_anonymize_ultrasound_logic.py`** (8 tests) - DICOM anonymization logic
- **`test_sample.py`** (5 tests) - Sample anonymization functions (in common/tests/)

### Core/Shared Tests (`tests/`) (32 tests)

- **`test_basic.py`** (8 tests) - Basic utility functions and core operations
- **`test_data_validation.py`** (7 tests) - Data validation and schema compliance
- **`test_configuration.py`** (8 tests) - Configuration management and utilities
- **`test_module_integration.py`** (11 tests) - Module integration and system functionality

## Test Coverage by Module

### AnnotateUltrasound Module Tests

- **Core Logic**: Annotation creation, rater management, data persistence
- **Cache Management**: Cache invalidation, state management, performance optimization
- **Reload Functionality**: Data reloading, state preservation, error handling
- **Overlay Management**: Overlay creation, rendering, visibility, and state management

### AnonymizeUltrasound Module Tests

- **Anonymization Logic**: Patient data anonymization, metadata handling
- **Sample Functions**: Basic anonymization operations and utilities

### Core/Shared Tests

- **Basic Utilities**: JSON handling, coordinate calculations, file operations
- **Data Validation**: Schema validation, coordinate validation, data consistency
- **Configuration**: Settings persistence, color management, file utilities
- **Module Integration**: Module loading, configuration, event handling

## Running Tests

### Quick Start - Run All Tests

```bash
# Use the provided test runner script (recommended)
python run_all_tests.py

# Or run manually from root directory
cd AnnotateUltrasound && python -m pytest tests/ -v
cd ../AnonymizeUltrasound && python -m pytest tests/ -v
cd .. && python -m pytest tests/ -v
python -m pytest AnonymizeUltrasound/common/tests/ -v
```

### Run Module-Specific Tests

```bash
# AnnotateUltrasound module tests (43 tests)
cd AnnotateUltrasound && python -m pytest tests/ -v

# AnonymizeUltrasound module tests (9 tests)
cd AnonymizeUltrasound && python -m pytest tests/ -v

# Core/shared tests (38 tests)
python -m pytest tests/ -v

# AnonymizeUltrasound common tests (5 tests)
python -m pytest AnonymizeUltrasound/common/tests/ -v
```

### Run Specific Test Categories

```bash
# Annotation logic and cache management
cd AnnotateUltrasound && python -m pytest tests/test_annotate_ultrasound_logic.py tests/test_cache_management.py -v

# Overlay and reload functionality
cd AnnotateUltrasound && python -m pytest tests/test_overlay_management.py tests/test_reload_functionality.py -v

# Data validation and configuration
python -m pytest tests/test_data_validation.py tests/test_configuration.py -v
```

### Run Legacy Integration Tests (Requires Slicer)

```bash
# Run within Slicer environment
python -m pytest AnnotateUltrasound/Testing/Python/ AnonymizeUltrasound/Testing/Python/ -v
```

## Test Categories

### 1. **Unit Tests** (Fast, No Dependencies)

- **Location**: `*/tests/test_*.py`
- **Execution Time**: ~0.5 seconds total
- **Dependencies**: Python standard library + pytest
- **Purpose**: Test core business logic and algorithms

### 2. **Integration Tests** (Slow, Requires Slicer)

- **Location**: `*/Testing/Python/`
- **Execution Time**: Could take several minutes
- **Dependencies**: Slicer environment
- **Purpose**: Test complete workflows and UI interactions

## Writing New Tests

### Module-Specific Tests

1. Create tests in the appropriate module directory: `[ModuleName]/tests/`
2. Follow the naming convention: `test_[functionality].py`
3. Use mock objects to avoid external dependencies
4. Test both success and failure scenarios

### Shared/Core Tests

1. Create tests in the main `tests/` directory for functionality that spans multiple modules
2. Focus on utilities, configuration, and integration functionality

### Test Structure

```python
class TestFeatureName:
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_object = MockClass()

    def test_specific_functionality(self):
        """Test description."""
        # Arrange
        # Act
        # Assert
```
