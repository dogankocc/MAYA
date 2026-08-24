@echo off
setlocal

set "AD=%~dp0..\android"

if exist "%AD%\gradlew.bat" (
  echo Gradle daemon durduruluyor...
  pushd "%AD%"
  call gradlew.bat --stop >nul 2>&1
  popd
)

timeout /t 2 /nobreak >nul
exit /b 0
