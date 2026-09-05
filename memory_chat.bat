@echo off
REM Thin launcher for Phase 5's memory_chat.py, so it's easy to find and
REM run from the project root without needing to remember its venv path
REM or cd into phaseFive first. The real script and its dependencies
REM (chromadb, its own venv) stay in phaseFive -- this just calls them
REM directly by full path, so it works no matter where it's launched from.
"%~dp0phaseFive\venv\Scripts\python.exe" "%~dp0phaseFive\memory_chat.py"
