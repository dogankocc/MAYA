@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0_build_paths.bat"
if errorlevel 1 exit /b 1

if not exist "%EXE%" (
  echo Build MAYA: scripts\build_server.bat
  exit /b 1
)

cd /d "%ROOT%"
py -3 "%~dp0migrate_corpus_to_jsonl.py"
py -3 "%~dp0label_corpus_intents.py"
call "%~dp0download_corpus.bat"
if errorlevel 1 exit /b 1

echo.
echo ===== TAM EGITIM (24 katman, Release) =====
echo Model: 24 layer, hidden 512, 21K+ ornek, auto steps.
echo Bu islem birkac saat surebilir — pencereyi kapatmayin.
echo.
"%EXE%" train --output model.ckpt --tokenizer tokenizer_data --corpus-dir data/corpus --corpus data/chat_corpus_tr.jsonl --steps auto --vocab 3072
if errorlevel 1 (
  echo.
  echo EGITIM BASARISIZ. Quantize atlaniyor.
  echo Once scripts\build_server.bat ile Release derleyip tekrar deneyin.
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
echo Hibrit ^(Ollama^): once VS Rebuild, sonra scripts\run_server_hybrid.bat
