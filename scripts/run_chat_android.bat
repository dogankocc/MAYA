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

start "LLM Server" "%ROOT%\scripts\run_server_hybrid.bat"
timeout /t 3 /nobreak >nul

cd /d "%ROOT%\chat_rn"
if not exist "node_modules\expo-build-properties" (
  echo Ilk kurulum: npm install
  call npm install
  if errorlevel 1 exit /b 1
)

echo.
echo Expo Go ile aciliyor (localhost modu — emulator icin daha stabil).
echo Ayri "LLM Chat" ikonu icin: scripts\run_chat_android_install.bat
call npx.cmd expo start --android --localhost --clear
