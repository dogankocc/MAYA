@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0_build_paths.bat"
if errorlevel 1 exit /b 1

if not exist "%EXE%" (
  echo Build MAYA in Visual Studio first.
  exit /b 1
)

start "LLM Server" "%EXE%" serve --backend local --model "%ROOT%\model.ckptq" --tokenizer "%ROOT%\tokenizer_data" --port 8765 --host 127.0.0.1
timeout /t 2 /nobreak >nul

set FLUTTER_BIN=%USERPROFILE%\flutter\bin\flutter.bat
if not exist "%FLUTTER_BIN%" set FLUTTER_BIN=flutter

set CHAT_EXE=%ROOT%\chat_app\build\windows\x64\runner\Release\llm_chat.exe
if exist "%CHAT_EXE%" (
  echo Hazir Windows uygulamasi aciliyor...
  start "" "%CHAT_EXE%"
  exit /b 0
)

if not exist "%USERPROFILE%\flutter\bin\flutter.bat" (
  echo Flutter yok. Kurmak icin: scripts\install_flutter.bat
  exit /b 1
)

cd /d "%ROOT%\chat_app"
"%FLUTTER_BIN%" run -d windows
