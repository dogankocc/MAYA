@echo off
setlocal

adb shell am force-stop com.llm.chat >nul 2>&1
timeout /t 1 /nobreak >nul
adb shell am start -a android.intent.action.VIEW -d "llm-chat://expo-development-client/?url=http://127.0.0.1:8081"
exit /b 0
