@echo off
setlocal

set "GP=%~dp0..\android\gradle.properties"
if not exist "%GP%" exit /b 0

findstr /C:"org.gradle.java.home" "%GP%" >nul 2>&1
if not errorlevel 1 exit /b 0

powershell -NoProfile -Command "$p='%GP%'; $c=Get-Content -Raw $p; if ($c -notmatch 'org.gradle.java.home') { Add-Content -Path $p -Value \"`norg.gradle.java.home=C\:/Program Files/Android/Android Studio/jbr`norg.gradle.java.installations.auto-download=false\" }"
exit /b 0
