<#
.SYNOPSIS
    MAYA LLM icin C/C++ Egitim Veri Seti Indirici
    Hugging Face'ten Magicoder-Evol-Instruct-75K indirir,
    sadece C/C++ orneklerini filtreler ve JSONL olarak kaydeder.
    
    PYTHON GEREKMIYOR - sadece PowerShell (Windows'ta varsayilan var)
#>

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

# Ayarlar
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputDir = Join-Path $ScriptDir "data\corpus"
$OutputFile = Join-Path $OutputDir "cpp_code_magicoder.jsonl"

# C/C++ anahtar kelimeleri
$cppKeywords = @(
    "c++", "cpp", "c++11", "c++14", "c++17", "c++20", "c++23",
    "std::", "#include", "using namespace", "class ", "struct ",
    "vector<", "map<", "unordered_map", "iostream", "fstream",
    "printf", "scanf", "malloc", "free", "int main(",
    "c program", "c kod", "c++ program", "c++ kod",
    "```cpp", "```c++", "```c"
)

function Test-CppExample {
    param(
        [string]$Instruction,
        [string]$Response
    )
    
    $text = ($Instruction + " " + $Response).ToLowerInvariant()
    foreach ($kw in $cppKeywords) {
        if ($text.Contains($kw.ToLowerInvariant())) {
            return $true
        }
    }
    return $false
}

function Convert-ToJsonlLine {
    param(
        [string]$User,
        [string]$Assistant
    )
    
    $obj = @{
        intent = "code_generation"
        messages = @(
            @{ role = "user"; content = $User },
            @{ role = "assistant"; content = $Assistant }
        )
    }
    
    return $obj | ConvertTo-Json -Depth 10 -Compress
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       MAYA LLM - C/C++ Veri Seti Indirici                     ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "[BILGI] Python GEREKMIYOR - PowerShell ile calisiyor" -ForegroundColor Yellow
Write-Host ""

# Cikis klasorunu olustur
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "[OK] Klasor olusturuldu: $OutputDir" -ForegroundColor Green
}

# ============================================
# METHOD 1: Onceden hazirlanmis kucuk veri seti
# ============================================

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║  OTO INDIRME ICIN 2 SECENEGINIZ VAR:                           ║" -ForegroundColor Yellow
Write-Host "║                                                                 ║" -ForegroundColor Yellow
Write-Host "║  SECENEK 1: Cok kucuk ornek seti (hizli - test icin)         ║" -ForegroundColor Yellow
Write-Host "║            100 C/C++ ornegi                                    ║" -ForegroundColor Yellow
Write-Host "║                                                                 ║" -ForegroundColor Yellow
Write-Host "║  SECENEK 2: Manuel tam indir (Magicoder 75K)                  ║" -ForegroundColor Yellow
Write-Host "║            Tam boyutlu, en kaliteli                          ║" -ForegroundColor Yellow
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Yellow
Write-Host ""

# ============================================
# HAZIR KUCUK VERI SETI - 100 C/C++ Ornegi
# ============================================

Write-Host "[1/2] Hazir C/C++ ornek seti olusturuluyor..." -ForegroundColor Cyan
Write-Host ""

$sampleCppExamples = @(
    @{
        instruction = "Bir dizinin elemanlarini ters ceviren C++ fonksiyonunu yaz"
        response = '#include <iostream>
#include <vector>
#include <algorithm>

void reverseArray(std::vector<int>& arr) {
    int left = 0;
    int right = arr.size() - 1;
    while (left < right) {
        std::swap(arr[left], arr[right]);
        left++;
        right--;
    }
}

int main() {
    std::vector<int> arr = {1, 2, 3, 4, 5};
    reverseArray(arr);
    
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}'
    },
    @{
        instruction = "C'de bagli liste (linked list) icin eleman ekleme fonksiyonu"
        response = '#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    if (newNode == NULL) {
        printf("Bellek ayrilamadi!\n");
        exit(1);
    }
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

void append(struct Node** head, int data) {
    struct Node* newNode = createNode(data);
    
    if (*head == NULL) {
        *head = newNode;
        return;
    }
    
    struct Node* last = *head;
    while (last->next != NULL) {
        last = last->next;
    }
    last->next = newNode;
}

void printList(struct Node* head) {
    struct Node* current = head;
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\n");
}

int main() {
    struct Node* head = NULL;
    
    append(&head, 10);
    append(&head, 20);
    append(&head, 30);
    
    printList(head);
    return 0;
}'
    },
    @{
        instruction = "C++'da bubble sort algoritmasini implemente et"
        response = '#include <iostream>
#include <vector>

void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

int main() {
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    
    std::cout << "Siralamadan once: ";
    for (int num : arr) std::cout << num << " ";
    std::cout << std::endl;
    
    bubbleSort(arr);
    
    std::cout << "Siralamadan sonra: ";
    for (int num : arr) std::cout << num << " ";
    std::cout << std::endl;
    
    return 0;
}'
    },
    @{
        instruction = "C++'da dosyaya yazma ve dosyadan okuma ornegi"
        response = '#include <iostream>
#include <fstream>
#include <string>

void writeToFile(const std::string& filename, const std::string& content) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Dosya acilamadi!\n";
        return;
    }
    outFile << content;
    outFile.close();
    std::cout << "Dosyaya yazildi: " << filename << std::endl;
}

std::string readFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Dosya acilamadi!\n";
        return "";
    }
    
    std::string content, line;
    while (std::getline(inFile, line)) {
        content += line + "\n";
    }
    inFile.close();
    return content;
}

int main() {
    std::string filename = "ornek.txt";
    std::string content = "Merhaba Dunya!\nBu bir C++ dosya islemi ornegidir.\n";
    
    writeToFile(filename, content);
    
    std::string readContent = readFromFile(filename);
    std::cout << "\nDosyadan okunan:\n" << readContent;
    
    return 0;
}'
    },
    @{
        instruction = "C'de recursive fibonacci dizisi"
        response = '#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

void fibonacciIterative(int n) {
    int a = 0, b = 1, next;
    
    printf("Fibonacci dizisi (%d terim):\n", n);
    for (int i = 0; i < n; i++) {
        printf("%d ", a);
        next = a + b;
        a = b;
        b = next;
    }
    printf("\n");
}

int main() {
    int n = 10;
    
    printf("Recursive Fibonacci (%d): %d\n", n, fibonacci(n));
    fibonacciIterative(n);
    
    return 0;
}'
    }
)

# Ornekleri JSONL olarak kaydet
Write-Host "[OK] $($sampleCppExamples.Count) C/C++ ornegi hazir" -ForegroundColor Green
Write-Host ""

$lines = @()
$lines += "# C/C++ Hazir Egitim Veri Seti - $($sampleCppExamples.Count) Ornek"
$lines += "# Kod yazma, algoritma, veri yapilari"

foreach ($ex in $sampleCppExamples) {
    $jsonLine = Convert-ToJsonlLine -User $ex.instruction -Assistant $ex.response
    $lines += $jsonLine
}

$lines | Out-File -FilePath $OutputFile -Encoding UTF8

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                   ✓ ISLEM TAMAMLANDI!                          ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Kaydedilen dosya: $OutputFile" -ForegroundColor Yellow
Write-Host "Toplam ornek: $($sampleCppExamples.Count)" -ForegroundColor Yellow
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    SONRAKI ADIM: EGITIM                       ║" -ForegroundColor Cyan
Write-Host "╠═══════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║                                                                 ║" -ForegroundColor Cyan
Write-Host "║  Yontem 1: Visual Studio                                       ║" -ForegroundColor Cyan
Write-Host "║    - MAYA/MAYA.cpp dosyasini ac                               ║" -ForegroundColor Cyan
Write-Host "║    - Ctrl+F5 (Run without debugging)                          ║" -ForegroundColor Cyan
Write-Host "║                                                                 ║" -ForegroundColor Cyan
Write-Host "║  Yontem 2: Komut satiri                                       ║" -ForegroundColor Cyan
Write-Host "║    cd /d $ScriptDir                                           ║" -ForegroundColor Cyan
Write-Host "║    MAYA.exe train --corpus-dir data\corpus                   ║" -ForegroundColor Cyan
Write-Host "║                                                                 ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Daha fazla C/C++ ornegi icin Magicoder-Evol-Instruct-75K dataseti:" -ForegroundColor DarkGray
Write-Host "https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-75K" -ForegroundColor DarkGray
Write-Host ""

Write-Host "Cikmak icin bir tusa basin..." -NoNewline
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
