@echo off
setlocal

set FLUTTER_ROOT=%USERPROFILE%\flutter

echo Flutter SDK: %FLUTTER_ROOT%
if exist "%FLUTTER_ROOT%\bin\flutter.bat" (
  echo Zaten kurulu.
  "%FLUTTER_ROOT%\bin\flutter.bat" --version
  exit /b 0
)

where git >nul 2>&1
if errorlevel 1 (
  echo Git gerekli. https://git-scm.com/download/win
  exit /b 1
)

echo Git ile indiriliyor (stable)...
git clone https://github.com/flutter/flutter.git -b stable --depth 1 "%FLUTTER_ROOT%"
if errorlevel 1 (
  echo Kurulum basarisiz.
  exit /b 1
)

echo Tamam. PATH: %FLUTTER_ROOT%\bin
"%FLUTTER_ROOT%\bin\flutter.bat" doctor -v
exit /b 0
