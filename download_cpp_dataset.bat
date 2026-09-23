@echo off
chcp 65001 >nul 2>&1
title MAYA - C/C++ Veri Seti Indirici

echo.
echo  ╔═══════════════════════════════════════════════════════════════╗
echo  ║                                                                   ║
echo  ║        MAYA LLM - C/C++ EGITIM VERI SETI INDIRICI             ║
echo  ║                                                                   ║
echo  ║  PYTHON GEREKMIYOR!                                            ║
echo  ║  Sadece PowerShell kullaniliyor (Windows'ta varsayilan var)   ║
echo  ║                                                                   ║
echo  ╚═══════════════════════════════════════════════════════════════╝
echo.

echo [KONTROL] PowerShell bulunuyor mu?
powershell -Command "exit 0" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [HATA] PowerShell bulunamadi!
    echo Windows 10/11'de varsayilan olarak gelir.
    echo.
    pause
    exit /b 1
)
echo [OK] PowerShell bulundu.

echo.
echo [CALISTIRILIYOR] PowerShell scripti...
echo Lutfen bekleyin...
echo.

cd /d "%~dp0"

PowerShell -ExecutionPolicy Bypass -File "%~dp0download_cpp_dataset.ps1"

pause
