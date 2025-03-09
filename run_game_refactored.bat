@echo off
call "%~dp0venv\Scripts\activate.bat"
set PYTHONPATH=%PYTHONPATH%;%~dp0
python -c "from ai_platform_trainer.core.launcher_refactored import main; main()"
