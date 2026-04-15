@echo off
title WeatherOracle - Setup Autostart
echo.
echo This will create a Windows Task Scheduler entry to start
echo WeatherOracle automatically when you log in.
echo.
echo Press Ctrl+C to cancel, or
pause

where python >nul 2>&1 && set "PYTHON=python" || (
    where python3 >nul 2>&1 && set "PYTHON=python3" || (
        echo ERROR: Python not found.
        pause
        exit /b 1
    )
)

set "SCRIPT_DIR=%~dp0"

schtasks /create /tn "WeatherOracle" /tr "\"%PYTHON%\" \"%SCRIPT_DIR%main.py\"" /sc onlogon /rl highest /f

if %errorlevel% equ 0 (
    echo.
    echo Autostart task created successfully.
    echo WeatherOracle will start when you log in.
) else (
    echo.
    echo Failed to create task. Try running this script as Administrator.
)
pause
