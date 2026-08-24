@echo off
setlocal

set MAX_TRIES=60
set TRY=0

:loop
set /a TRY+=1
if %TRY% GTR %MAX_TRIES% (
  echo Metro 60 sn icinde hazir olmadi.
  exit /b 1
)

curl -s http://127.0.0.1:8081/status 2>nul | findstr /C:"packager-status:running" >nul
if %ERRORLEVEL%==0 (
  echo Metro hazir.
  exit /b 0
)

timeout /t 2 /nobreak >nul
goto loop
