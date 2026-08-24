@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0_build_paths.bat"
if errorlevel 1 exit /b 1

if not exist "%EXE%" (
  echo CMakeProject2.exe bulunamadi. Once Visual Studio ile derleyin.
  exit /b 1
)

if not exist "%ROOT%\model.ckptq" (
  echo model.ckptq bulunamadi. Once scripts\prepare_model.bat calistirin.
  exit /b 1
)

cd /d "%ROOT%"
echo Sunucu baslatiliyor...
echo Yerel model: %ROOT%\model.ckptq
echo Ollama icin: ollama pull llama3.2  ^(OpenAI backend secildiginde^)

REM Eski exe --model'i checkpoint saniyor; llama3.2 BURAYA YAZILMAMALI.
"%EXE%" serve --backend local --model "%ROOT%\model.ckptq" --tokenizer "%ROOT%\tokenizer_data" --port 8765 --host 127.0.0.1
