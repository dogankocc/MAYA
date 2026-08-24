@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0set_java_home.bat" >nul 2>&1

echo Emulator uyandiriliyor...
adb shell input keyevent KEYCODE_WAKEUP >nul 2>&1

call "%ROOT%\chat_rn\scripts\adb_reverse.bat"

set APK=%ROOT%\chat_rn\android\app\build\outputs\apk\debug\app-debug.apk
if exist "%APK%" (
  echo LLM Chat APK yeniden kuruluyor...
  adb install -r "%APK%"
) else (
  echo APK yok. Once: scripts\run_chat_android_install.bat
)

adb shell am force-stop host.exp.exponent >nul 2>&1
adb shell am force-stop com.llm.chat >nul 2>&1
timeout /t 2 /nobreak >nul
call "%~dp0launch_chat_app.bat"

echo.
echo Metro kapaliysa ayri terminalde:
echo   cd chat_rn
echo Metro: scripts\start_metro.bat
