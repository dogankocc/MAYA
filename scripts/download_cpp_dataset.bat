@echo off
chcp 65001 >nul 2>&1
title MAYA - C/C++ Egitim Veri Seti Indirici

echo.
echo  ╔═══════════════════════════════════════════════════════════════╗
echo  ║                                                                   ║
echo  ║        MAYA LLM - C/C++ EGITIM VERI SETI INDIRICI              ║
echo  ║                                                                   ║
echo  ║  Bu script otomatik olarak:                                      ║
echo  ║  1. Gerekli Python kutuphanesini yukler (datasets)              ║
echo  ║  2. Hugging Face'den Magicoder-Evol-Instruct-75K indirir       ║
echo  ║  3. Sadece C ve C++ orneklerini filtreler                       ║
echo  ║  4. Projenin JSONL formatina cevirir                             ║
echo  ║  5. data/corpus/ klasorune kaydeder                              ║
echo  ║                                                                   ║
echo  ╚═══════════════════════════════════════════════════════════════╝
echo.

echo [KONTROL] Python bulunuyor mu?
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [HATA] Python yuklu degil!
    echo Lutfen Python 3.8+ yukleyin: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo [OK] Python bulundu.

echo.
echo [1/3] Gerekli kutuphane yukleniyor (datasets)...
python -m pip install datasets --quiet 2>nul
if errorlevel 1 (
    echo [UYARI] Otomatik yukleme basarisiz, elle yuklemeye calisiliyor...
    python -m pip install datasets
)
echo [OK] Hazir.

echo.
echo [2/3] Veri seti indiriliyor ve C/C++ filtreleniyor...
echo       Bu islem internet hizina gore 2-10 dakika surebilir.
echo       Lutfen bekleyin...
echo.

cd /d "%~dp0"
python scripts/download_cpp_dataset.py

if errorlevel 1 (
    echo.
    echo.
    echo ╔═══════════════════════════════════════════════════════════════╗
    echo ║                        [HATA]                                    ║
    echo ║           Indirme islemi basarisiz oldu!                       ║
    echo ╚═══════════════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

echo.
echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                                                                   ║
echo ║                    ✓ ISLEM TAMAMLANDI!                          ║
echo ║                                                                   ║
echo ║  C/C++ egitim veri seti hazir!                                  ║
echo ║                                                                   ║
echo ║  Sonraki adim: EGITIMI BASLAT                                   ║
echo ║                                                                   ║
echo ║  Method 1: Visual Studio ile                                    ║
echo ║            - MAYA/MAYA.cpp dosyasini ac                         ║
echo ║            - Ctrl+F5 veya Run butonuna bas                     ║
echo ║                                                                   ║
echo ║  Method 2: Komut satiri ile                                     ║
echo ║            cd /d %~dp0                                          ║
echo ║            MAYA.exe train --corpus-dir data/corpus             ║
echo ║                                                                   ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.
echo Devam etmek icin bir tusa basin ve EGITIMI BASLATIN...
pause >nul
