@echo off
setlocal enabledelayedexpansion

:: Default to running all tests
SET TEST_ARGS=

:: Check for command line arguments
IF "%1" == "unit" (
    SET TEST_ARGS=-m unit
    echo Running only unit tests...
) ELSE IF "%1" == "integration" (
    SET TEST_ARGS=-m integration
    echo Running only integration tests...
) ELSE IF "%1" == "coverage" (
    SET TEST_ARGS=--cov=ai_platform_trainer --cov-report=html
    echo Running tests with coverage report...
) ELSE IF "%1" == "verbose" (
    SET TEST_ARGS=-v
    echo Running tests with verbose output...
) ELSE IF "%1" == "help" (
    echo Usage: run_tests.bat [option]
    echo Options:
    echo   unit      - Run only unit tests
    echo   integration - Run only integration tests
    echo   coverage  - Generate coverage report
    echo   verbose   - Show detailed test output
    echo   help      - Show this help message
    echo   (no option) - Run all tests
    exit /b 0
) ELSE (
    echo Running all tests...
)

:: Activate virtual environment if it exists
IF EXIST venv\Scripts\activate.bat (
    CALL venv\Scripts\activate.bat
)

:: Run the tests
pytest %TEST_ARGS%

:: If coverage report was generated, show the path
IF "%1" == "coverage" (
    echo Coverage report generated at reports\coverage_html\index.html
)

:: Deactivate virtual environment if it was activated
IF EXIST venv\Scripts\deactivate.bat (
    CALL venv\Scripts\deactivate.bat
)

endlocal
