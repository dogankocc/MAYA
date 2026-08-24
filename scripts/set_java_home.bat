@echo off
setlocal EnableDelayedExpansion

if defined JAVA_HOME (
  if exist "%JAVA_HOME%\bin\java.exe" (
    exit /b 0
  )
)

set "JBR="
for %%P in (
  "%LOCALAPPDATA%\Programs\Android Studio\jbr"
  "C:\Program Files\Android\Android Studio\jbr"
  "C:\Program Files\Microsoft\Android Studio\jbr"
) do (
  if exist "%%~P\bin\java.exe" set "JBR=%%~P"
)

if not defined JBR (
  echo Java bulunamadi. Android Studio kurulu olmali.
  echo Android Studio -^> Settings -^> Gradle -^> JDK yolunu kontrol edin.
  exit /b 1
)

set "JAVA_HOME=%JBR%"
set "PATH=%JAVA_HOME%\bin;%PATH%"
if not exist "C:\gradle" mkdir "C:\gradle" >nul 2>&1
set "GRADLE_USER_HOME=C:\gradle"
echo JAVA_HOME=%JAVA_HOME%
echo GRADLE_USER_HOME=%GRADLE_USER_HOME%
exit /b 0
