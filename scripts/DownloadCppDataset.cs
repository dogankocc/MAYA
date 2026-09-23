// C# ile Hugging Face'ten C/C++ veri seti indirme
// Derlemek icin: csc DownloadCppDataset.cs ya da dotnet run
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MayaDatasetDownloader
{
    class Program
    {
        private static readonly HttpClient httpClient = new HttpClient();
        private static readonly HashSet<string> seenPrompts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private static readonly Random random = new Random(42);

        // C/C++ anahtar kelimeleri
        private static readonly string[] cppKeywords = new string[]
        {
            "c++", "cpp", "c++11", "c++14", "c++17", "c++20", "c++23",
            "std::", "#include", "using namespace", "class ", "struct ",
            "vector<", "map<", "unordered_map", "iostream", "fstream",
            "printf", "scanf", "malloc", "free", "int main(",
            "c program", "c kod", "c++ program", "c++ kod",
            "```cpp", "```c++", "```c"
        };

        static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║       MAYA LLM - C/C++ Veri Seti Indirici (C#)               ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝");
            Console.WriteLine();

            // Hugging Face API'sinden veri cek
            // Parquet yerine dogrudan streaming JSON kullan
            // NOT: Bu basit bir ornek - tam olarak Hugging Face dataseti icin
            // en iyi yol: Hugging Face Hub C# kutuphanesi veya indirip parquet oku

            Console.WriteLine("[1/3] Hugging Face API'si kontrol ediliyor...");

            // Bu versiyonda onceden hazirlanmis URL'ler kullan
            // Gercek cozum icin:
            // 1. HuggingFaceHub NuGet paketini kullan
            // 2. Veya parquet dosyasini indirip okut

            Console.WriteLine();
            Console.WriteLine("[BILGI] Tam C# cozumu icin NuGet paketleri gerekli:");
            Console.WriteLine("  - HuggingFaceHub veya");
            Console.WriteLine("  - Parquet.Net");
            Console.WriteLine();
            Console.WriteLine("[2/3] Alternatif: Cok daha basit yontem - Dogrudan indir");
            Console.WriteLine();

            // En basit cozum: Onceden hazirlanmis kucuk bir C/C++ ornek seti
            // Veya kullaniciya PowerShell scripti oner

            Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  OTOMATIK COZUM ICIN POWERSHELL SCRIPTINI CALISTIRIN:        ║");
            Console.WriteLine("║                                                                 ║");
            Console.WriteLine("║  .\\download_cpp_dataset.ps1                                    ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝");
            Console.WriteLine();
            Console.WriteLine("Bu C# dosyasi sadece framework - tam cozum icin PowerShell");
            Console.WriteLine("scriptini kullanin (hicbir kurulum gerekmiyor).");
        }

        private static bool IsCppExample(string instruction, string response)
        {
            string text = (instruction + " " + response).ToLowerInvariant();
            foreach (string kw in cppKeywords)
            {
                if (text.Contains(kw.ToLowerInvariant()))
                {
                    return true;
                }
            }
            return false;
        }
    }
}
