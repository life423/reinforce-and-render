@echo off
set PYTHONPATH=%PYTHONPATH%;%~dp0venv\Lib\site-packages
python -m ai_platform_trainer.main
