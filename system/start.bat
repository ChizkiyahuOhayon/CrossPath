@echo off
REM Windows: double-click this file to set up and start the demo.
REM First run creates the virtualenv, installs dependencies, builds the
REM catalog and trains the two endpoint heads + gate; later runs just start
REM the server in a couple of seconds.
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (
    python bootstrap.py %*
    goto :end
)
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 bootstrap.py %*
    goto :end
)
echo Python was not found. Install Python 3.9+ from https://www.python.org/downloads/
pause
:end
pause
