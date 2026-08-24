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

set EXPO_NO_GIT_STATUS=1



echo npm install...

call npm install

if errorlevel 1 exit /b 1



echo Expo native moduller...

call npx.cmd expo install expo-constants expo-asset expo-font expo-splash-screen

if errorlevel 1 exit /b 1



if /I "%CLEAN_ANDROID%"=="1" (

  echo Native proje temiz yenileniyor...

  call scripts\stop_gradle.bat

  call npx.cmd expo prebuild --platform android --clean

  if errorlevel 1 (

    call scripts\unlock_android.bat

    if errorlevel 1 exit /b 1

    call npx.cmd expo prebuild --platform android --clean

    if errorlevel 1 exit /b 1

  )

) else if not exist "android\app\build.gradle" (

  echo Native proje olusturuluyor...

  call npx.cmd expo prebuild --platform android

  if errorlevel 1 exit /b 1

) else (

  echo Native proje mevcut, prebuild atlaniyor. Temiz yenileme: set CLEAN_ANDROID=1

)



call scripts\patch_gradle_jdk.bat



echo.

echo Emulatore "LLM Chat" kuruluyor (5-10 dk)...

call npx.cmd expo run:android --no-bundler

if errorlevel 1 exit /b 1



echo Metro baslatiliyor...

call npx.cmd expo start --android --localhost --clear

