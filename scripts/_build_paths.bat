@echo off
REM Tek kaynak: tum scriptler her zaman Release (x64-release) kullanir.
if not defined ROOT (
  echo HATA: _build_paths.bat icin once ROOT tanimlanmali.
  exit /b 1
)
set BUILD_PRESET=x64-release
set CMAKE_CONFIG=Release
set BUILD_DIR=%ROOT%\out\build\x64-release
set EXE=%BUILD_DIR%\CMakeProject2\CMakeProject2.exe
