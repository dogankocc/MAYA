@echo off
setlocal

set ROOT=%~dp0..\chat_rn
cd /d "%ROOT%"

echo Metro baslatiliyor (localhost:8081)...
echo PowerShell yerine CMD kullanilir — execution policy sorunu yok.
echo.
echo LLM Chat ikonunu acmadan once bu pencere acik kalsin.
echo Kapatmak icin Ctrl+C

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js bulunamadi.
  exit /b 1
)

call "%ROOT%\scripts\adb_reverse.bat"

REM npx.ps1 yerine npx.cmd — PowerShell script engeli yok
call npx.cmd expo start --clear
