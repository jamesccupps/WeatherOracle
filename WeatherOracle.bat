@echo off
title WeatherOracle v2.1
cd /d "%~dp0"

:: Try common Python locations
where python >nul 2>&1 && (
    set "PYTHON=python"
    goto :found
)
where python3 >nul 2>&1 && (
    set "PYTHON=python3"
    goto :found
)

echo ERROR: Python not found. Install Python 3.10+ from python.org
pause
exit /b 1

:found
echo Using Python: %PYTHON%
%PYTHON% -c "import requests, sklearn, numpy; print('All dependencies OK')" 2>nul || (
    echo Installing dependencies...
    %PYTHON% -m pip install -r requirements.txt
)

echo Starting WeatherOracle...
%PYTHON% main.py
pause
