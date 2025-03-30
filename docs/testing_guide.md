# Testing Guide for AI Platform Trainer

This document provides guidelines for testing the AI Platform Trainer application, explains the existing test structure, and offers best practices for adding new tests.

## Table of Contents

- [Testing Guide for AI Platform Trainer](#testing-guide-for-ai-platform-trainer)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Test Structure](#test-structure)
  - [Running Tests](#running-tests)
  - [Writing New Tests](#writing-new-tests)
  - [Mocking](#mocking)
  - [Test Coverage](#test-coverage)

## Overview

The AI Platform Trainer uses pytest as its testing framework. The test suite is organized into several categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **Performance Tests**: Test performance characteristics (to be implemented)

The test suite aims to verify that:
- Individual components work correctly
- Components interact as expected
- The game mechanics function as intended
- The AI models produce reasonable outputs
- Performance remains acceptable

## Test Structure

The tests are organized to mirror the structure of the application:

```
tests/
│
├── unit/                  # Unit tests for isolated components
│   ├── entities/          # Tests for game entities
│   │   ├── behaviors/     # Tests for entity behaviors
│   │   └── components/    # Tests for entity components
│   ├── engine/            # Tests for engine components
│   │   ├── core/          # Tests for core engine functionality
│   │   ├── physics/       # Tests for physics engine
│   │   └── rendering/     # Tests for rendering engine
│   └── ai/                # Tests for AI models
│       ├── models/        # Tests for model definitions
│       └── training/      # Tests for training functionality
│
├── integration/           # Integration tests between components
│   ├── test_game_mechanics.py  # Tests for game mechanics
│   └── test_training_pipeline.py  # Tests for training pipeline
│
└── performance/           # Tests for performance characteristics
```

## Running Tests

To run all tests, use the provided scripts:

**Windows:**
```
run_tests.bat
```

**Linux/Mac:**
```
./run_tests.sh
```

You can also run specific test categories:

```bash
# Run only unit tests
python -m pytest tests/unit/ -v

# Run only integration tests
python -m pytest tests/integration/ -v

# Run tests by keyword
python -m pytest -k "missile" -v

# Run with coverage report
python -m pytest --cov=ai_platform_trainer tests/
```

## Writing New Tests

When adding new tests, follow these guidelines:

1. **Test Location**: Place the test in the appropriate directory that mirrors the structure of the code being tested.
2. **Naming Convention**: Use `test_` as a prefix for all test files and test methods.
3. **Fixtures**: Use pytest fixtures to set up common test objects.
4. **Independence**: Tests should be independent of each other and not rely on state from previous tests.
5. **Coverage**: Aim to test both the normal operation and edge cases.

Here's a basic template for a test file:

```python
"""
Test description - what's being tested and why.
"""
import pytest
from unittest.mock import Mock, patch

from ai_platform_trainer.path.to.module import ComponentToTest


@pytest.fixture
def component():
    """Create a component for testing."""
    return ComponentToTest(param1="value1", param2="value2")


class TestComponentName:
    """Tests for ComponentToTest."""
    
    def test_normal_operation(self, component):
        """Test that the component works correctly under normal conditions."""
        # Arrange
        input_value = "test input"
        expected_output = "expected result"
        
        # Act
        actual_output = component.method_to_test(input_value)
        
        # Assert
        assert actual_output == expected_output
    
    def test_edge_case(self, component):
        """Test that the component handles edge cases correctly."""
        # Test implementation
```

## Mocking

Since the game relies on Pygame and other external dependencies, mocking is essential for testing. Use the `unittest.mock` module for mocking:

```python
# Mock pygame.Surface
with patch('pygame.Surface') as mock_surface:
    mock_surface.return_value = Mock()
    # Test code that uses pygame.Surface
    
# Mock time-dependent functions
with patch('time.time') as mock_time:
    mock_time.return_value = 1000  # Fixed time for deterministic tests
    # Test code that uses time.time()
```

For more complex components like the AI models, create mock objects that mimic their behavior:

```python
# Mock AI model
model = Mock()
model.return_value = torch.tensor([0.5, 0.5])  # Mock output
component_under_test = ComponentToTest(model=model)
```

## Test Coverage

Aim for comprehensive test coverage of the codebase. Priority areas include:

1. **Enemy AI Controller**: The core logic for enemy movement, especially the mechanisms that prevent freezing.
2. **Missile AI Controller**: Logic for guiding missiles toward targets.
3. **Collision Detection**: Ensure collisions are correctly detected and handled.
4. **Player Controls**: Verify player input is correctly processed.
5. **Game Loop**: Test the main game loop and state transitions.

When adding new features, include tests that verify the feature works as expected in isolation (unit tests) and integrates correctly with the rest of the system (integration tests).

For performance-critical components, consider adding performance tests that verify the component meets performance expectations.
