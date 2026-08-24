@echo off
setlocal

echo Proje surecleri kapatiliyor...

taskkill /F /IM CMakeProject2.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1

if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" (
  "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" reverse --remove-all >nul 2>&1
)

set "GRADLE_HOME=C:\gradle"
if exist "%GRADLE_HOME%" (
  if exist "c:\Users\dogan.koc\source\repos\CMakeProject2\chat_rn\android\gradlew.bat" (
    pushd "c:\Users\dogan.koc\source\repos\CMakeProject2\chat_rn\android"
    set GRADLE_USER_HOME=C:\gradle
    call gradlew.bat --stop >nul 2>&1
    popd
  )
)

echo.
echo Kapatildi: CMakeProject2, Metro (node), adb reverse, Gradle daemon.
echo Emulator acik kaldi — kapatmak icin emulator penceresini kapat.
echo.
echo Sifirdan baslat: scripts\start_fresh.bat
