@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo Creating virtual environment in .venv ...
  where py >nul 2>nul && py -3 -m venv .venv
)
if not exist "%VENV_PY%" (
  where python >nul 2>nul && python -m venv .venv
)
if not exist "%VENV_PY%" (
  echo.
  echo [ERROR] Cannot create .venv. Install Python 3 from python.org and tick "Add to PATH",
  echo         or ensure the "py" launcher works. Then double-click this file again.
  echo.
  pause
  exit /b 1
)

echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :pip_fail

echo Installing dependencies ...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_fail

if not exist ".env" (
  if exist ".env.example" (
    copy /y ".env.example" ".env" >nul
    echo Created .env from template. Edit .env to add your API keys, then run this again.
    pause
    exit /b 0
  )
)

echo Starting Streamlit ...
"%VENV_PY%" -m streamlit run ui\app.py
if errorlevel 1 (
  echo.
  echo [ERROR] Streamlit failed. Read the message above.
  echo.
  pause
  exit /b 1
)
goto :eof

:pip_fail
echo.
echo [ERROR] pip failed. Check your network / proxy and try again.
echo.
pause
exit /b 1
