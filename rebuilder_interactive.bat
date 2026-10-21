@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem Interactive PyInstaller builder. Place next to .py, .ico and optional Tk hook.
set "WINDOWED=1"
set "SHORTCUT_DIR=%~dp0"
cd /d "%~dp0" || goto failed
if "%SHORTCUT_DIR:~-1%"=="\" set "SHORTCUT_DIR=%SHORTCUT_DIR:~0,-1%"

echo.
echo Available Python scripts:
set /a COUNT=0
for /f "delims=" %%F in ('dir /b /a-d /on "*.py" 2^>nul') do (
    if /i not "%%F"=="rebuilder_tk_icon_hook.py" (
        set /a COUNT+=1
        set "FILE_!COUNT!=%%F"
        echo   !COUNT!^) %%F
    )
)
if !COUNT! EQU 0 (
    echo No Python scripts found in %~dp0
    goto failed
)

:select_script
set "SELECTION="
set /p "SELECTION=Choose script number (1-!COUNT!): "
set "PY_FILE="
for /l %%I in (1,1,!COUNT!) do if "!SELECTION!"=="%%I" set "PY_FILE=!FILE_%%I!"
if not defined PY_FILE (
    echo Invalid selection. Try again.
    goto select_script
)
for %%F in ("!PY_FILE!") do set "DEFAULT_NAME=%%~nF"
echo.
echo Selected: !PY_FILE!
set "EXE_NAME="
set /p "EXE_NAME=Executable name [!DEFAULT_NAME!]: "
if not defined EXE_NAME set "EXE_NAME=!DEFAULT_NAME!"

rem Validate without CMD metacharacters embedded in regex strings.
python -c "import os,sys; n=os.environ['EXE_NAME']; forbidden=set(map(chr,[34,42,47,58,60,62,63,92,124])); reserved=['CON','PRN','AUX','NUL']+['COM'+str(i) for i in range(1,10)]+['LPT'+str(i) for i in range(1,10)]; sys.exit(int(not n or any(ord(c)<32 or c in forbidden for c in n) or n.endswith(('.', ' ')) or n.split('.')[0].upper() in reserved))"
if errorlevel 1 (
    echo Invalid executable name, or Python is not available.
    goto failed
)

echo.
echo Building !PY_FILE! as !EXE_NAME!.exe
set "ICON=%~1"
if not defined ICON for %%F in (*.ico) do if not defined ICON set "ICON=%%F"
set "ICONARG="
set "ICONDATAARG="
set "HOOKARG="
if defined ICON (
    if not exist "!ICON!" (
        echo Icon not found: !ICON!
        goto failed
    )
    echo Using icon: !ICON!
    set "ICONARG=--icon "!ICON!""
    if exist "%~dp0rebuilder_tk_icon_hook.py" (
        set "ICONDATAARG=--add-data "!ICON!;_rebuilder_assets""
        set "HOOKARG=--runtime-hook "%~dp0rebuilder_tk_icon_hook.py""
    ) else (
        echo NOTE: No Tkinter hook found. EXE icon only.
    )
) else (
    echo No .ico file found; using default executable icon.
)
rem Bundle sibling-project helpers that are loaded by file path at runtime,
rem plus pyserial, which PyInstaller cannot discover through them.
set "SIBLINGARG="
if exist "%~dp0..\VoltageTuner\board.py" set "SIBLINGARG=!SIBLINGARG! --add-data "%~dp0..\VoltageTuner\board.py;VoltageTuner""
if exist "%~dp0..\ThorlabsControl\ArduinoSampler.py" set "SIBLINGARG=!SIBLINGARG! --add-data "%~dp0..\ThorlabsControl\ArduinoSampler.py;ThorlabsControl""
if defined SIBLINGARG set "SIBLINGARG=!SIBLINGARG! --hidden-import serial"
set "MODEARG=--console"
if "%WINDOWED%"=="1" set "MODEARG=--windowed"

rem Check running executable before attempting to overwrite it.
tasklist /fi "imagename eq !EXE_NAME!.exe" | find /i "!EXE_NAME!.exe" >nul
if not errorlevel 1 (
    echo The executable is running. Close it before rebuilding.
    goto failed
)

echo.
echo Checking PyInstaller...
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    python -m pip install pyinstaller
    if errorlevel 1 goto failed
)

echo.
echo Starting build. Full output will be saved to rebuilder_build.log
rem Do not group this command in parentheses: it contains user-selected paths.
python -m PyInstaller --noconfirm --clean --onefile !MODEARG! --name "!EXE_NAME!" !ICONARG! !ICONDATAARG! !HOOKARG! !SIBLINGARG! "!PY_FILE!" >"%~dp0rebuilder_build.log" 2>&1
if errorlevel 1 (
    echo.
    echo PyInstaller failed. Last 30 lines of the log:
    powershell -NoProfile -Command "Get-Content -LiteralPath $env:CD\rebuilder_build.log -Tail 30"
    goto failed
)

set "TARGET=%~dp0dist\!EXE_NAME!.exe"
set "WORKDIR=%~dp0"
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $env:SHORTCUT_DIR ($env:EXE_NAME+'.lnk'))); $s.TargetPath=$env:TARGET; $s.WorkingDirectory=$env:WORKDIR.TrimEnd('\'); $s.IconLocation=$env:TARGET+',0'; $s.Save()"
if errorlevel 1 goto failed

echo.
echo SUCCESS: !TARGET!
echo Shortcut: !SHORTCUT_DIR!\!EXE_NAME!.lnk
echo Log: %~dp0rebuilder_build.log
goto finished

:failed
echo.
echo BUILD FAILED. See messages above and rebuilder_build.log if present.
:finished
echo.
pause
exit /b
