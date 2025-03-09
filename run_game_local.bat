@echo off
call "%~dp0venv\Scripts\activate.bat"
set PYTHONPATH=%PYTHONPATH%;%~dp0
python -m ai_platform_trainer.main
