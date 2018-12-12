@echo off

@rem Change the working directory to the location of this file so that relative paths will work
cd /D "%~dp0"

@rem Make sure the environment variables are up-to-date. This is useful if the user installed python a moment ago.
call ./RefreshEnv.cmd

setlocal EnableDelayedExpansion

@rem Run the is_safe_to_upgrade function and save the output to a temp file.
python -c "from rlbot.utils import public_utils; print(public_utils.is_safe_to_upgrade());" > %temp%\is_safe_to_upgrade.txt

IF %ERRORLEVEL% NEQ 0 (
    @rem The python command failed, so rlbot is probably not installed at all. Safe to 'upgrade'.
    set is_safe_to_upgrade=True
) ELSE (
    @rem read the file containing the python output.
    set /p is_safe_to_upgrade= < %temp%\is_safe_to_upgrade.txt
)
del %temp%\is_safe_to_upgrade.txt

IF "!is_safe_to_upgrade!"=="True" (
    python -m pip install -r requirements.txt --upgrade
) ELSE (
    echo Will not attempt to upgrade rlbot because files are in use.
)

rem the last structural thing in this batch is a switch case
IF "%1" == "hastam" goto HASTAM_CASE
IF "%1" == "croissant" goto LECROISSANT_CASE
IF "%1" == "baguette" goto LEBAGUETTE_CASE
IF "%1" == "record" goto RECORD_CASE

goto DEFAULT_CASE

:RECORD_CASE
	python -c "from LeFramework.runner import custom_main; custom_main('./rlbot_recorder.cfg');"
goto END_CASE

:LECROISSANT_CASE
	python -c "from LeFramework.runner import custom_main; custom_main('./rlbot_LeCroissant.cfg');"
goto END_CASE

:LEBAGUETTE_CASE
	python -c "from LeFramework.runner import custom_main; custom_main('./rlbot_LeBaguette.cfg');"
goto END_CASE

:HASTAM_CASE
	python -c "from LeFramework.runner import custom_main; custom_main('./rlbot_HastamRevenge.cfg');"
goto END_CASE

:DEFAULT_CASE
python -c "from rlbot import runner; runner.main();"

:END_CASE
pause
