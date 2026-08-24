@echo off
setlocal

set ROOT=%~dp0..

call "%~dp0stop_all.bat"
timeout /t 2 /nobreak >nul

echo.
echo ===== SIFIRDAN BASLATILIYOR =====
echo.

call "%~dp0ensure_emulator.bat"
if errorlevel 1 exit /b 1

call "%~dp0set_java_home.bat"
if errorlevel 1 exit /b 1

call "%ROOT%\chat_rn\scripts\adb_reverse.bat"

start "LLM Server" "%ROOT%\scripts\run_server_hybrid.bat"
timeout /t 3 /nobreak >nul

start "Metro" "%ROOT%\scripts\start_metro.bat"
echo Metro baslatildi, hazir olmasi bekleniyor...
call "%~dp0wait_metro.bat"
if errorlevel 1 exit /b 1

set APK=%ROOT%\chat_rn\android\app\build\outputs\apk\debug\app-debug.apk
if exist "%APK%" (
  echo APK kuruluyor...
  adb install -r "%APK%"
)

echo Ilk bundle icin 45 sn bekleniyor (Metro penceresinde "Android Bundled")...
timeout /t 45 /nobreak >nul

adb shell am force-stop com.llm.chat >nul 2>&1
timeout /t 1 /nobreak >nul
call "%~dp0launch_chat_app.bat"

echo.
echo Hazir. Emulatorda LLM Chat acildi.
echo Hala siyahsa Metro penceresinde "r" tusuna bas veya uygulamayi kapat-ac.
echo Sunucu: http://127.0.0.1:8765  Metro: http://localhost:8081
