@echo off
title WeatherOracle Web Display
cd /d "%~dp0"

where python >nul 2>&1 && set "PYTHON=python" || (
    where python3 >nul 2>&1 && set "PYTHON=python3" || (
        echo ERROR: Python not found.
        pause
        exit /b 1
    )
)

echo Starting web display on http://localhost:8847
%PYTHON% weather_display.py
pause
