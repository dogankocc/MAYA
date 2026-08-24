@echo off
setlocal

call "%~dp0stop_gradle.bat"

set "AD=%~dp0..\android"
if not exist "%AD%" exit /b 0

echo Android klasoru kilidi aciliyor...
for /L %%i in (1,1,5) do (
  rmdir /S /Q "%AD%" 2>nul
  if not exist "%AD%" exit /b 0
  call "%~dp0stop_gradle.bat"
  timeout /t 2 /nobreak >nul
)

echo.
echo android klasoru kilitli. Android Studio / Explorer kapat, sonra tekrar dene.
exit /b 1
