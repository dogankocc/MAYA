@echo off
setlocal

set ROOT=%~dp0..
call "%~dp0_build_paths.bat"
if errorlevel 1 exit /b 1

set VSDEV="C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"

call "%~dp0build_server.bat"
if errorlevel 1 exit /b 1

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

echo.
echo Unit testler derleniyor (%BUILD_PRESET%)...
"%CMAKE%" --build "%BUILD_DIR%" --target llm_core_tests --config %CMAKE_CONFIG%
if errorlevel 1 exit /b 1

echo.
echo Testler calistiriliyor...
"%CMAKE%" --build "%BUILD_DIR%" --target test --config %CMAKE_CONFIG%
exit /b %ERRORLEVEL%
