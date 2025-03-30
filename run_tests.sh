#!/bin/bash
echo "Running all tests with pytest..."
source venv/bin/activate && python -m pytest tests/ -v
