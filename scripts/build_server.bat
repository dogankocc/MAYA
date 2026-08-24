@echo off

setlocal



set ROOT=%~dp0..

call "%~dp0_build_paths.bat"

if errorlevel 1 exit /b 1



set VSDEV="C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"



echo LLM sunucusu (Release) derleniyor...

echo NOT: chat_app / llm_chat / Flutter DEGIL.



tasklist /FI "IMAGENAME eq MAYA.exe" 2>nul | find /I "MAYA.exe" >nul

if not errorlevel 1 (

  echo Calisan MAYA.exe kapatiliyor...

  taskkill /F /IM MAYA.exe >nul 2>&1

  timeout /t 1 /nobreak >nul

)



if not exist %VSDEV% (

  set VSDEV="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

)

if not exist %VSDEV% (

  echo VsDevCmd bulunamadi.

  exit /b 1

)



call %VSDEV% -no_logo -arch=amd64



set CMAKE=

for %%P in (

  "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

  "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

) do if exist %%P set CMAKE=%%~P



if not defined CMAKE (

  echo cmake.exe bulunamadi.

  exit /b 1

)



if not exist "%BUILD_DIR%\build.ninja" (

  echo Ilk Release CMake yapilandirmasi: %BUILD_PRESET%

  "%CMAKE%" --preset %BUILD_PRESET%

  if errorlevel 1 exit /b 1

)



"%CMAKE%" --build "%BUILD_DIR%" --target MAYA --config %CMAKE_CONFIG%

if errorlevel 1 (

  echo Derleme basarisiz.

  exit /b 1

)



if not exist "%EXE%" (

  echo Derleme bitti ama exe bulunamadi: %EXE%

  exit /b 1

)



echo.

echo OK: %EXE%

"%EXE%" version 2>nul

exit /b 0

