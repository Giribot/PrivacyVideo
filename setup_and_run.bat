@echo off
REM ---------------------------------------------------------------------------
REM setup_and_run.bat — Windows 11, ASCII‑only
REM ---------------------------------------------------------------------------
REM 1. Create a virtual environment (".venv") in the current folder
REM 2. Install dependencies from requirements.txt
REM 3. Run the main script "privacyVideo.py"
REM ---------------------------------------------------------------------------
REM Usage:
REM   setup_and_run.bat                      (uses python.exe from PATH)
REM   setup_and_run.bat C:\Python311\python.exe   (specific interpreter)
REM   Any further arguments are passed to privacyVideo.py.
REM ---------------------------------------------------------------------------

SETLOCAL ENABLEDELAYEDEXPANSION

:: -------- 0) Select Python executable ---------------------------------------
IF "%~1"=="" (
    SET "PYTHON=python"
) ELSE (
    SET "PYTHON=%~1"
    SHIFT
)

SET "VENV_DIR=.venv"

:: -------- 1) Create venv if missing ----------------------------------------
IF NOT EXIST "%VENV_DIR%" (
    ECHO [+] Creating virtual environment "%VENV_DIR%" ...
    "%PYTHON%" -m venv "%VENV_DIR%"
    IF ERRORLEVEL 1 (
        ECHO [!] Failed to create virtual environment
        EXIT /B 1
    )
)

:: -------- 2) Activate venv --------------------------------------------------
CALL "%VENV_DIR%\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    ECHO [!] Failed to activate virtual environment
    EXIT /B 1
)

:: -------- 3) Upgrade pip ----------------------------------------------------
ECHO [+] Upgrading pip ...
python -m pip install --upgrade pip

:: -------- 4) Install requirements ------------------------------------------
IF NOT EXIST "requirements.txt" (
    ECHO [!] requirements.txt not found
    EXIT /B 1
)
ECHO [+] Installing dependencies from requirements.txt ...
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO [!] Failed to install dependencies
    EXIT /B 1
)

:: -------- 5) Run the main script -------------------------------------------
IF NOT EXIST "privacyVideo.py" (
    ECHO [!] privacyVideo.py not found
    EXIT /B 1
)
ECHO [+] Running praivacyVideo.py ...
python privacyVideo.py %*

ENDLOCAL
