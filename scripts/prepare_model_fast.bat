@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0_build_paths.bat"
if errorlevel 1 exit /b 1

set PROFILE=%ROOT%\data\corpus\profile_dogan.jsonl

if not exist "%EXE%" (
  echo Once scripts\build_server.bat calistirin.
  exit /b 1
)

if not exist "%PROFILE%" (
  echo Profil dosyasi bulunamadi: %PROFILE%
  exit /b 1
)

cd /d "%ROOT%"

echo.
echo ===== HIZLI EGITIM (temel sohbet + profil, 5 katman, Release) =====
echo Veri: yerlesik sohbet + data\corpus\profile_dogan.jsonl
echo 1000 step, vocab 1024. Tam egitim: scripts\prepare_model.bat
echo.

"%EXE%" train --output model.ckpt --tokenizer tokenizer_data --corpus-dir none --corpus data/corpus/profile_dogan.jsonl --steps 1000 --vocab 1024
if errorlevel 1 (
  echo EGITIM BASARISIZ.
  exit /b 1
)

if not exist "model.ckpt" (
  echo model.ckpt olusturulmadi.
  exit /b 1
)

echo Quantizing...
"%EXE%" quantize --input model.ckpt --output model.ckptq
if errorlevel 1 exit /b 1

echo.
echo Tamam. Sunucu: scripts\run_server.bat

