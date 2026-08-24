@echo off
setlocal

set ROOT=%~dp0..
cd /d "%ROOT%"

where py >nul 2>&1
if errorlevel 1 (
  echo Python bulunamadi. py launcher veya Python 3 kurun.
  exit /b 1
)

echo Turkce egitim verisi indiriliyor (Hugging Face - kapsamli setler)...
py -3 "%ROOT%\scripts\download_training_corpus.py"
if errorlevel 1 (
  echo Indirme basarisiz. Internet baglantinizi kontrol edin.
  exit /b 1
)

echo Tamam. Veri: data\corpus\ (JSONL + intent)
py -3 "%ROOT%\scripts\label_corpus_intents.py"
