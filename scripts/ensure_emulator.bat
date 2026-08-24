@echo off
setlocal EnableDelayedExpansion

REM Emulator aciksa dokunma; kapaliysa baslat. Asla restart etmez.

if defined ANDROID_SDK_ROOT (
  set "SDK=%ANDROID_SDK_ROOT%"
) else if defined ANDROID_HOME (
  set "SDK=%ANDROID_HOME%"
) else (
  set "SDK=%LOCALAPPDATA%\Android\Sdk"
)

set "ADB=%SDK%\platform-tools\adb.exe"
set "EMULATOR=%SDK%\emulator\emulator.exe"

if not exist "%ADB%" (
  echo adb bulunamadi: %ADB%
  echo Android SDK platform-tools kurulu olmali.
  exit /b 1
)

if not exist "%EMULATOR%" (
  echo emulator bulunamadi: %EMULATOR%
  echo Android Studio -^> SDK Manager -^> Android Emulator kurun.
  exit /b 1
)

set "EMULATOR_RUNNING=0"
"%ADB%" devices 2>nul | findstr /I /C:"emulator-" | findstr /I /C:"device" >nul
if not errorlevel 1 (
  set "EMULATOR_RUNNING=1"
  for /f "tokens=1" %%I in ('"%ADB%" devices 2^>nul ^| findstr /I "emulator-" ^| findstr /I "device"') do (
    echo Emulator zaten acik: %%I
  )
)

if "!EMULATOR_RUNNING!"=="1" goto wait_boot

if defined ANDROID_AVD (
  set "AVD=%ANDROID_AVD%"
) else (
  set "AVD="
  for /f "usebackq delims=" %%A in (`"%EMULATOR%" -list-avds 2^>nul`) do (
    if not defined AVD set "AVD=%%A"
  )
)

if not defined AVD (
  echo Sanal cihaz ^(AVD^) bulunamadi. Android Studio Device Manager'dan olusturun.
  exit /b 1
)

echo Emulator kapali — baslatiliyor: !AVD!
start "Android Emulator" "%EMULATOR%" -avd "!AVD!"
echo Emulator penceresi acilmasi biraz surebilir...

:wait_boot
"%ADB%" wait-for-device >nul 2>&1

set /a BOOT_TRIES=0
:boot_poll
set "BOOT_DONE="
for /f "delims=" %%B in ('"%ADB%" shell getprop sys.boot_completed 2^>nul') do set "BOOT_DONE=%%B"
if "!BOOT_DONE!"=="1" goto boot_ready
set /a BOOT_TRIES+=1
if !BOOT_TRIES! GEQ 90 (
  echo Emulator boot zaman asimi ^(~3 dk^). Yine de devam ediliyor...
  goto boot_ready
)
timeout /t 2 /nobreak >nul
goto boot_poll

:boot_ready
echo Emulator hazir.
exit /b 0
