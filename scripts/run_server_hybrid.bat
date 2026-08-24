@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0_build_paths.bat"
if errorlevel 1 exit /b 1

if not exist "%EXE%" (
  echo MAYA.exe bulunamadi.
  exit /b 1
)

cd /d "%ROOT%"
"%EXE%" version 2>nul | findstr /C:"hybrid-backend" >nul
if errorlevel 1 (
  echo Hibrit sunucu icin once Visual Studio'da Rebuild Solution yapin.
  echo Simdilik: scripts\run_server.bat ^(yerel model^)
  exit /b 1
)

echo Hibrit sunucu baslatiliyor...
"%EXE%" serve --backend openai --model-path "%ROOT%\model.ckptq" --tokenizer "%ROOT%\tokenizer_data" --api-host 127.0.0.1:11434 --openai-model llama3.2 --port 8765 --host 127.0.0.1
