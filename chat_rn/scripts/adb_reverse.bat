@echo off
setlocal EnableDelayedExpansion

if defined ANDROID_SDK_ROOT (
  set "SDK=%ANDROID_SDK_ROOT%"
) else if defined ANDROID_HOME (
  set "SDK=%ANDROID_HOME%"
) else (
  set "SDK=%LOCALAPPDATA%\Android\Sdk"
)

set "ADB=%SDK%\platform-tools\adb.exe"
if not exist "%ADB%" exit /b 0

"%ADB%" devices 2>nul | findstr /I "emulator-" | findstr /I "device" >nul
if errorlevel 1 exit /b 0

echo Emulator ag yonlendirme (Metro + sunucu)...
"%ADB%" reverse --remove-all >nul 2>&1
"%ADB%" reverse tcp:8081 tcp:8081 >nul 2>&1
"%ADB%" reverse tcp:8765 tcp:8765 >nul 2>&1
exit /b 0
