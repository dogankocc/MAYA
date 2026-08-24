@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0_build_paths.bat"
if errorlevel 1 exit /b 1

if not exist "%EXE%" (
  echo Once: scripts\build_server.bat
  exit /b 1
)

call "%~dp0ensure_emulator.bat"
if errorlevel 1 exit /b 1

call "%~dp0set_java_home.bat"
if errorlevel 1 exit /b 1

call "%ROOT%\chat_rn\scripts\adb_reverse.bat"

tasklist /FI "IMAGENAME eq node.exe" 2>nul | find /I "node.exe" >nul
if errorlevel 1 (
  echo Metro baslatiliyor (ayri pencere)...
  start "Metro" "%ROOT%\scripts\start_metro.bat"
  timeout /t 8 /nobreak >nul
) else (
  echo Metro zaten calisiyor olabilir.
)

tasklist /FI "IMAGENAME eq MAYA.exe" 2>nul | find /I "MAYA.exe" >nul
if errorlevel 1 (
  start "LLM Server" "%ROOT%\scripts\run_server_hybrid.bat"
  timeout /t 3 /nobreak >nul
)

set APK=%ROOT%\chat_rn\android\app\build\outputs\apk\debug\app-debug.apk
if exist "%APK%" (
  echo APK kuruluyor...
  adb install -r "%APK%"
)

adb shell am force-stop com.llm.chat >nul 2>&1
timeout /t 1 /nobreak >nul
call "%~dp0launch_chat_app.bat"

echo.
echo LLM Chat acildi. Siyah ekran olursa Metro penceresinde bundle bitsin, sonra uygulamayi kapat-ac.
echo Metro: scripts\start_metro.bat
