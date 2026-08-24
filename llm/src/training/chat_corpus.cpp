#include "llm/training/chat_corpus.hpp"

#include "llm/training/jsonl_parser.hpp"
#include "llm/training/dialogue_format.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace llm::training {

namespace {

std::string Trim(std::string value) {
  const auto isSpace = [](unsigned char ch) { return std::isspace(ch) != 0; };
  while (!value.empty() && isSpace(static_cast<unsigned char>(value.front()))) {
    value.erase(value.begin());
  }
  while (!value.empty() && isSpace(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  return value;
}

Result<DialogueSample> ParseLine(const std::string& line) {
  const std::size_t separator = line.find('|');
  if (separator == std::string::npos) {
    return Result<DialogueSample>::Fail(ErrorCode::InvalidArgument, "invalid dialogue line");
  }

  DialogueSample sample{
      .prompt = Trim(line.substr(0, separator)),
      .response = Trim(line.substr(separator + 1)),
  };

  if (sample.prompt.empty() || sample.response.empty()) {
    return Result<DialogueSample>::Fail(ErrorCode::InvalidArgument, "empty prompt or response");
  }

  return Result<DialogueSample>::Ok(std::move(sample));
}

void AppendSample(std::vector<DialogueSample>& samples, const char* prompt, const char* response) {
  samples.push_back(DialogueSample{.prompt = prompt, .response = response});
}

} // namespace

Result<std::vector<DialogueSample>> LoadDialogueCorpus(const std::string& path) {
  namespace fs = std::filesystem;

  std::ifstream input(path);
  if (!input.is_open()) {
    return Result<std::vector<DialogueSample>>::Fail(ErrorCode::IoError, "corpus file not found: " + path);
  }

  const bool jsonlFormat = fs::path(path).extension() == ".jsonl";
  std::vector<DialogueSample> samples;
  std::string line;
  while (std::getline(input, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    if (jsonlFormat) {
      const auto parsed = ParseJsonlDialogueLine(line);
      if (!parsed.IsOk()) {
        return Result<std::vector<DialogueSample>>::Fail(parsed.GetError().code, parsed.GetError().message);
      }
      samples.push_back(parsed.Value());
      continue;
    }

    const auto parsed = ParseLine(line);
    if (!parsed.IsOk()) {
      return Result<std::vector<DialogueSample>>::Fail(parsed.GetError().code, parsed.GetError().message);
    }
    samples.push_back(parsed.Value());
  }

  if (samples.empty()) {
    return Result<std::vector<DialogueSample>>::Fail(ErrorCode::InvalidArgument, "corpus file has no samples");
  }

  return Result<std::vector<DialogueSample>>::Ok(std::move(samples));
}

Result<std::vector<DialogueSample>> LoadDialogueCorpusDirectory(const std::string& directoryPath) {
  namespace fs = std::filesystem;

  std::error_code errorCode;
  if (!fs::exists(directoryPath, errorCode) || !fs::is_directory(directoryPath, errorCode)) {
    return Result<std::vector<DialogueSample>>::Fail(ErrorCode::IoError,
                                                     "corpus directory not found: " + directoryPath);
  }

  std::vector<DialogueSample> samples;
  std::vector<fs::path> files;
  for (const fs::directory_entry& entry : fs::directory_iterator(directoryPath, errorCode)) {
    if (errorCode) {
      return Result<std::vector<DialogueSample>>::Fail(ErrorCode::Internal, errorCode.message());
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".jsonl") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end());
  for (const fs::path& file : files) {
    const auto loaded = LoadDialogueCorpus(file.string());
    if (!loaded.IsOk()) {
      continue;
    }
    const auto& extra = loaded.Value();
    samples.insert(samples.end(), extra.begin(), extra.end());
  }

  if (samples.empty()) {
    return Result<std::vector<DialogueSample>>::Fail(ErrorCode::InvalidArgument,
                                                     "corpus directory has no valid samples: " + directoryPath);
  }

  return Result<std::vector<DialogueSample>>::Ok(std::move(samples));
}

std::vector<DialogueSample> MergeDialogueCorpora(std::vector<DialogueSample> base,
                                                 const std::vector<DialogueSample>& extra) {
  std::unordered_set<std::string> seen;
  seen.reserve(base.size() + extra.size());
  for (const DialogueSample& sample : base) {
    seen.insert(sample.prompt);
  }

  for (const DialogueSample& sample : extra) {
    if (seen.insert(sample.prompt).second) {
      base.push_back(sample);
    }
  }

  return base;
}

std::vector<DialogueSample> DefaultDialogueCorpus() {
  std::vector<DialogueSample> samples;
  samples.reserve(120);

  AppendSample(samples, "merhaba", "Merhaba! Bugun size nasil yardimci olabilirim?");
  AppendSample(samples, "selam", "Selam! Sorularinizi bekliyorum.");
  AppendSample(samples, "nasilsin", "Iyiyim, tesekkur ederim. Siz nasilsiniz?");
  AppendSample(samples, "naber", "Iyidir, buradayim. Sen nasilsin?");
  AppendSample(samples, "gunaydin", "Gunaydin! Umarim guzel bir gun gecirirsiniz.");
  AppendSample(samples, "iyi aksamlar", "Iyi aksamlar! Size nasil destek olabilirim?");
  AppendSample(samples, "tesekkurler", "Rica ederim, her zaman sorabilirsiniz.");
  AppendSample(samples, "sagol", "Ne demek, yardimci olabildiysem ne mutlu.");
  AppendSample(samples, "gorusuruz", "Gorusmek uzere! Iyi gunler dilerim.");
  AppendSample(samples, "hosca kal", "Hosca kalin, tekrar beklerim.");
  AppendSample(samples, "kimsin", "Ben yerel bir sohbet asistaniyim. Sorularinizi yanitlamak icin buradayim.");
  AppendSample(samples, "ne yapabilirsin", "Sorularinizi yanitlayabilir, kisa aciklamalar yapabilir ve sohbet edebilirim.");
  AppendSample(samples, "bana yardim et", "Tabii, hangi konuda yardim istersiniz?");
  AppendSample(samples, "anlamadim", "Sorun degil, isterseniz daha basit bir sekilde tekrar anlatirim.");
  AppendSample(samples, "tekrar eder misin", "Elbette, ayni cevabi daha net bir sekilde tekrar yazabilirim.");
  AppendSample(samples, "bugun nasil hissediyorsun", "Ben bir programim ama size yardim etmeye hazirim.");
  AppendSample(samples, "sikildim", "Biraz sohbet edebiliriz. Isterseniz yeni bir konu secelim.");
  AppendSample(samples, "yorgunum", "Dinlenmek iyi gelir. Kisa bir mola vermenizi onerebilirim.");
  AppendSample(samples, "mutluyum", "Harika! Guzel haberleri paylasmak her zaman guzel.");
  AppendSample(samples, "uzgunum", "Uzuldugunuzu duydum. Isterseniz konusabiliriz.");
  AppendSample(samples, "stresliyim", "Derin nefes almak ve kisa molalar stresi azaltmaya yardimci olabilir.");
  AppendSample(samples, "python nedir", "Python, okunmasi kolay ve yaygin kullanilan bir programlama dilidir.");
  AppendSample(samples, "react native nedir", "React Native, JavaScript ile mobil uygulama gelistirmeyi saglayan bir cercevedir.");
  AppendSample(samples, "llm nedir", "LLM, buyuk dil modeli demektir ve metin uretip anlayabilir.");
  AppendSample(samples, "api nedir", "API, yazilimlarin birbiriyle iletisim kurmasini saglayan arayuzdur.");
  AppendSample(samples, "veritabani nedir", "Veritabani, verilerin duzenli saklandigi ve sorgulandigi sistemdir.");
  AppendSample(samples, "hata aliyorum", "Hatayi paylasir misiniz? Birlikte adim adim cozebiliriz.");
  AppendSample(samples, "kod calismiyor", "Once hata mesajini kontrol edelim, sonra olasi nedeni bulalim.");
  AppendSample(samples, "bug ne demek", "Bug, yazilimdaki hata veya beklenmeyen davranis anlamina gelir.");
  AppendSample(samples, "test neden onemli", "Testler, kodun dogru calistigini ve degisikliklerin bozmadigini gosterir.");
  AppendSample(samples, "git nedir", "Git, kod versiyonlarini takip eden bir versiyon kontrol sistemidir.");
  AppendSample(samples, "commit ne demek", "Commit, kodda yaptiginiz degisikliklerin kaydedilmesidir.");
  AppendSample(samples, "sunucu calismiyor", "Once sunucunun acik oldugunu ve dogru portu dinledigini kontrol edin.");
  AppendSample(samples, "baglanti yok", "Ag adresini, portu ve sunucunun calisip calismadigini kontrol edelim.");
  AppendSample(samples, "model nedir", "Model, veriden ogrenen ve tahmin ya da metin ureten yapay zeka sistemidir.");
  AppendSample(samples, "egitim ne kadar surer", "Model boyutuna ve veri miktariina gore dakikalardan saatlere kadar degisebilir.");
  AppendSample(samples, "chatgpt gibi olur mu", "Kucuk yerel modeller ChatGPT kadar guclu olmaz ama temel sohbet yapabilir.");
  AppendSample(samples, "daha iyi cevap icin ne yapmaliyim", "Modeli daha fazla ornek ve daha uzun sure egitmek kaliteyi artirir.");
  AppendSample(samples, "turkce konus", "Tabii, bundan sonra Turkce devam edebiliriz.");
  AppendSample(samples, "ingilizce konus", "Sure, we can continue in English if you prefer.");
  AppendSample(samples, "kisa cevap ver", "Tamam, daha kisa ve net yanitlar verecegim.");
  AppendSample(samples, "detayli anlat", "Elbette, konuyu adim adim ve daha ayrintili aciklayabilirim.");
  AppendSample(samples, "ornek ver", "Ornegin bir REST API, GET ve POST istekleriyle veri alip gonderebilir.");
  AppendSample(samples, "adim adim anlat", "Once hedefi belirleyelim, sonra sirayla uygulayalim.");
  AppendSample(samples, "basitce anlat", "Basitce soylemek gerekirse, sistem bir soruya metinle cevap uretir.");
  AppendSample(samples, "fark ne", "Fark, iki seyin birbirinden ayiran temel ozelligidir.");
  AppendSample(samples, "avantaji ne", "Avantaji, isi daha hizli ve duzenli yapmanizi saglamasidir.");
  AppendSample(samples, "dezavantaji ne", "Dezavantaji, kucuk modellerin sinirli bilgiye sahip olmasidir.");
  AppendSample(samples, "onerin ne", "Once temel kurulumu tamamlayin, sonra modeli daha fazla veriyle egitin.");
  AppendSample(samples, "hangi dili onerirsin", "Amaciniza gore degisir; mobil icin React Native, sunucu icin C++ veya Python uygun olabilir.");
  AppendSample(samples, "android icin ne onerirsin", "Android icin React Native veya Kotlin ile gelistirme yapabilirsiniz.");
  AppendSample(samples, "windows icin ne onerirsin", "Windows icin C++ veya C# ile masaustu uygulamasi gelistirebilirsiniz.");
  AppendSample(samples, "proje nasil calisir", "Bu projede C++ sunucu modeli calistirir, mobil arayuz HTTP ile baglanir.");
  AppendSample(samples, "sunucu ne ise yarar", "Sunucu, modeli calistirir ve uygulamadan gelen mesajlara cevap uretir.");
  AppendSample(samples, "tokenizer nedir", "Tokenizer, metni modele uygun parcalara ayiran bilesendir.");
  AppendSample(samples, "temperature nedir", "Temperature, model cevabinin ne kadar rastgele olacagini belirler.");
  AppendSample(samples, "greedy mod nedir", "Greedy mod, her adimda en olasi kelimeyi secerek daha tutarli cevap verir.");
  AppendSample(samples, "max token nedir", "Max token, modelin tek seferde uretebilecegi maksimum cevap uzunlugudur.");
  AppendSample(samples, "reset ne yapar", "Reset, sohbet baglamini sifirlar ve yeni bir konusma baslatir.");
  AppendSample(samples, "merhaba nasilsin", "Merhaba! Iyiyim, tesekkurler. Siz nasilsiniz?");
  AppendSample(samples, "selam nasilsin", "Selam! Iyiyim, size nasil yardimci olabilirim?");
  AppendSample(samples, "yardima ihtiyacim var", "Tabii, hangi konuda yardima ihtiyaciniz var?");
  AppendSample(samples, "acil yardim", "Buradayim. Sorununuzu kisaca yazin, hemen yardimci olayim.");
  AppendSample(samples, "plan yap", "Once hedefi yazalim, sonra kucuk adimlara bolelim.");
  AppendSample(samples, "motivasyon ver", "Kucuk adimlarla ilerlemek bile buyuk sonuclar dogurabilir.");
  AppendSample(samples, "espri yap", "Bir programci neden gec kaldi? Cunku sonsuz donguye girmisti.");
  AppendSample(samples, "hikaye anlat", "Kisa bir hikaye: Sabırla calisan gelistirici, sonunda calisan bir proje cikardi.");
  AppendSample(samples, "siir yaz", "Kod satirlari gece boyunca, sabah calisan bir uygulama dogar.");
  AppendSample(samples, "bugun ne ogrenmeliyim", "Bugun temel bir kavram secip kucuk bir ornek uygulamak iyi baslangictir.");
  AppendSample(samples, "kariyer tavsiyesi", "Temel becerileri saglam ogrenip duzenli pratik yapmak cok etkilidir.");
  AppendSample(samples, "zaman yonetimi", "Isleri onceliklendirip kucuk zaman bloklarina bolmek verimliligi artirir.");
  AppendSample(samples, "odaklanamiyorum", "Telefonu uzaklastirip 25 dakikalik kisa calisma araliklari deneyebilirsiniz.");
  AppendSample(samples, "uyuyamiyorum", "Yatmadan once ekran suresini azaltmak ve duzenli saatler yardimci olabilir.");
  AppendSample(samples, "spor onerisi", "Gunluk kisa yuruyusler hem beden hem zihin icin faydalidir.");
  AppendSample(samples, "kahve icmeli miyim", "Olculu tuketim cogu kisi icin sorun olmaz, ama cok fazla uyku bozabilir.");
  AppendSample(samples, "hava nasil", "Canli hava durumuna erisimim yok, ama hava durumu uygulamalarina bakabilirsiniz.");
  AppendSample(samples, "saat kac", "Guncel saati sistem saatinizden kontrol edebilirsiniz.");
  AppendSample(samples, "hangi gun", "Bugunun tarihini cihazinizin takviminden gorebilirsiniz.");
  AppendSample(samples, "istanbul nerede", "Istanbul, Turkiye'nin kuzeybatisinda yer alan buyuk bir sehirdir.");
  AppendSample(samples, "ankara nedir", "Ankara, Turkiye'nin baskentidir.");
  AppendSample(samples, "turkiye nerede", "Turkiye, Asya ve Avrupa kitalarinda topraklari olan bir ulkedir.");
  AppendSample(samples, "matematik zor", "Matematik pratikle kolaylasir; kucuk adimlarla ogrenmek en iyisidir.");
  AppendSample(samples, "2 arti 2 kac", "2 arti 2 esittir 4.");
  AppendSample(samples, "10 bolu 2 kac", "10 bolu 2 esittir 5.");
  AppendSample(samples, "karekok 16 kac", "16'nin karekoku 4'tur.");
  AppendSample(samples, "pi sayisi nedir", "Pi sayisi yaklasik 3.14159 degerindedir.");
  AppendSample(samples, "fotosentez nedir", "Fotosentez, bitkilerin isikla besin uretmesidir.");
  AppendSample(samples, "su neden onemli", "Su, vucudun ve ekosistemin saglikli calismasi icin gereklidir.");
  AppendSample(samples, "elektrik nedir", "Elektrik, yuklu parcaciklarin hareketiyle olusan enerji bicimidir.");
  AppendSample(samples, "internet nasil calisir", "Internet, birbirine bagli cihazlarin veri paketleri gondermesiyle calisir.");
  AppendSample(samples, "wifi nedir", "WiFi, kablosuz ag baglantisi saglayan teknolojidir.");
  AppendSample(samples, "bluetooth nedir", "Bluetooth, kisa mesafede cihazlar arasi kablosuz iletisim saglar.");
  AppendSample(samples, "sifre nasil olmali", "Guclu sifre uzun olmali, harf, rakam ve sembol icermelidir.");
  AppendSample(samples, "guvenlik onerisi", "Guncel yazilim kullanin, sifreleri paylasmayin, iki adimli dogrulama acin.");
  AppendSample(samples, "veri yedegi", "Onemli dosyalari duzenli yedeklemek veri kaybini onler.");
  AppendSample(samples, "performans artirma", "Gereksiz islemleri azaltmak ve onbellekleme performansi artirabilir.");
  AppendSample(samples, "cache nedir", "Cache, sik kullanilan veriyi hizli erisim icin gecici saklayan bellek katmanidir.");
  AppendSample(samples, "thread nedir", "Thread, ayni program icinde paralel calisabilen is parcacigidir.");
  AppendSample(samples, "async nedir", "Async, islemlerin beklemeden arka planda surmesini saglayan yaklasimdir.");
  AppendSample(samples, "json nedir", "JSON, verileri metin olarak saklayan yaygin bir veri formatidir.");
  AppendSample(samples, "http nedir", "HTTP, web uzerinde veri alisverisi icin kullanilan protokoldur.");
  AppendSample(samples, "rest nedir", "REST, web servislerinde kaynaklara HTTP ile erisim saglayan mimaridir.");
  AppendSample(samples, "docker nedir", "Docker, uygulamalari konteyner icinde calistirmayi saglar.");
  AppendSample(samples, "kubernetes nedir", "Kubernetes, konteynerleri otomatik yoneten bir orchestration sistemidir.");
  AppendSample(samples, "sql nedir", "SQL, veritabanlarinda veri sorgulamak icin kullanilan dildir.");
  AppendSample(samples, "nosql nedir", "NoSQL, farkli veri modelleri kullanan veritabani turleridir.");
  AppendSample(samples, "algoritma nedir", "Algoritma, bir problemi cozmek icin adim adim yontemdir.");
  AppendSample(samples, "veri yapisi nedir", "Veri yapisi, verilerin bellekte duzenlenme bicimidir.");
  AppendSample(samples, "linked list nedir", "Linked list, elemanlarin birbirine bagli oldugu zincir yapisidir.");
  AppendSample(samples, "binary search nedir", "Binary search, sirali dizide hizli arama yapan yontemdir.");
  AppendSample(samples, "big o nedir", "Big O, algoritmanin calisma maliyetini ifade eder.");
  AppendSample(samples, "machine learning nedir", "Machine learning, veriden ogrenen algoritmalar alanidir.");
  AppendSample(samples, "deep learning nedir", "Deep learning, cok katmanli sinir aglariyla ogrenme yapan yontemdir.");
  AppendSample(samples, "overfitting nedir", "Overfitting, modelin egitim verisine fazla uyum saglayip genelleme yapamamasidir.");
  AppendSample(samples, "dataset nedir", "Dataset, model egitiminde kullanilan veri kumesidir.");
  AppendSample(samples, "epoch nedir", "Epoch, egitim verisinin tamaminin bir kez kullanilmasidir.");
  AppendSample(samples, "learning rate nedir", "Learning rate, model agirliklarinin ne kadar hizli guncellenecegini belirler.");
  AppendSample(samples, "loss nedir", "Loss, modelin ne kadar hata yaptigini olcen degerdir.");
  AppendSample(samples, "optimizer nedir", "Optimizer, loss'a gore model agirliklarini gunceller.");
  AppendSample(samples, "checkpoint nedir", "Checkpoint, egitim sirasinda kaydedilen model durumudur.");
  AppendSample(samples, "quantization nedir", "Quantization, model agirliklarini daha dusuk hassasiyette saklayarak hiz kazandirir.");
  AppendSample(samples, "inference nedir", "Inference, egitilmis modelle tahmin veya metin uretme asamasidir.");
  AppendSample(samples, "prompt nedir", "Prompt, modele verilen giris metnidir.");
  AppendSample(samples, "context nedir", "Context, modelin cevap uretirken dikkate aldigi onceki metindir.");
  AppendSample(samples, "hallucination nedir", "Hallucination, modelin gercege uymayan bilgi uretmesidir.");
  AppendSample(samples, "guvenilir mi", "Yerel kucuk modeller hata yapabilir; onemli bilgileri mutlaka dogrulayin.");
  AppendSample(samples, "kaynak goster", "Bu demo model egitim verisinden ogrendigi kadarini kullanir, dis kaynak vermez.");
  AppendSample(samples, "bilmiyorum de", "Emin olmadigim konularda tahmin yapmak yerine bilmedigimi soylemeliyim.");
  AppendSample(samples, "fikrini soyle", "Bence once basit bir cozum denemek, sonra iyilestirmek en mantiklisi.");
  AppendSample(samples, "hangisi daha iyi", "Hangisinin daha iyi oldugu kullanim amacina gore degisir.");
  AppendSample(samples, "evet", "Tamam, devam edelim.");
  AppendSample(samples, "hayir", "Anladim, baska bir konuya gecebilir veya durabiliriz.");
  AppendSample(samples, "tamam", "Harika, bir sonraki adima gecebiliriz.");
  AppendSample(samples, "peki", "Peki, devam ediyorum.");
  AppendSample(samples, "devam et", "Devam ediyorum, bir sonraki konuyu acikliyorum.");
  AppendSample(samples, "dur", "Tamam, durdum. Hazir oldugunuzda devam ederiz.");
  AppendSample(samples, "baska soru", "Tabii, yeni sorunuzu yazabilirsiniz.");
  AppendSample(samples, "son bir sey", "Elbette, son sorunuzu dinliyorum.");
  AppendSample(samples, "cok tesekkurler", "Rica ederim, yardimci olabildiysem ne mutlu.");
  AppendSample(samples, "harikasin", "Tesekkur ederim, siz de cok naziksiniz.");
  AppendSample(samples, "kotu cevap", "Haklisiniz, isterseniz daha iyi bir yanit vermeye calisayim.");
  AppendSample(samples, "yanlis anladin", "Ozur dilerim, lutfen tekrar aciklayin, daha iyi anlayayim.");
  AppendSample(samples, "daha net yaz", "Tabii, daha net ve anlasilir sekilde yaziyorum.");
  AppendSample(samples, "ornek kod yaz", "Ornek: print('Merhaba') ifadesi ekrana Merhaba yazar.");
  AppendSample(samples, "c++ ogrenmek istiyorum", "C++ ogrenmek icin once temel degiskenler, donguler ve fonksiyonlarla baslayin.");
  AppendSample(samples, "javascript ogrenmek istiyorum", "JavaScript icin once degiskenler, fonksiyonlar ve DOM ile baslayabilirsiniz.");
  AppendSample(samples, "react ogrenmek istiyorum", "React icin once component, state ve props kavramlarini ogrenmek iyi baslangictir.");
  AppendSample(samples, "mobil uygulama nasil yapilir", "Once arayuz tasarimi, sonra backend baglantisi ve test adimlari izlenir.");
  AppendSample(samples, "backend nedir", "Backend, uygulamanin sunucu tarafinda calisan is mantigi ve veri katmanidir.");
  AppendSample(samples, "frontend nedir", "Frontend, kullanicinin gordugu ve etkilesime girdigi arayuz katmanidir.");
  AppendSample(samples, "full stack nedir", "Full stack, hem frontend hem backend gelistirebilen yaklasimdir.");
  AppendSample(samples, "debug nasil yapilir", "Once hatayi tekrarlayin, loglari okuyun, sonra kucuk parcalara ayirip test edin.");
  AppendSample(samples, "log ne ise yarar", "Loglar, uygulamanin calisirken ne yaptigini anlamaya yardim eder.");
  AppendSample(samples, "unit test nedir", "Unit test, kodun en kucuk parcalarinin dogru calistigini kontrol eder.");
  AppendSample(samples, "integration test nedir", "Integration test, birden fazla bilesenin birlikte dogru calistigini kontrol eder.");
  AppendSample(samples, "ci cd nedir", "CI/CD, kod degisikliklerinin otomatik test ve dagitim surecidir.");
  AppendSample(samples, "agile nedir", "Agile, kucuk adimlarla hizli gelistirme yapan proje yonetim yaklasimidir.");
  AppendSample(samples, "scrum nedir", "Scrum, Agile cercevesinde sprintlerle calisan ekip modelidir.");
  AppendSample(samples, "product owner kim", "Product owner, urunun ne yapacagina karar veren roldur.");
  AppendSample(samples, "sprint nedir", "Sprint, belirli surede tamamlanacak islerin kisa gelistirme dongusudur.");
  AppendSample(samples, "deadline yaklasti", "Oncelikleri netlestirip en kritik isleri once tamamlamak iyi olur.");
  AppendSample(samples, "proje yetismiyor", "Kapsami kucultup en onemli ozelliklere odaklanmayi dusunun.");
  AppendSample(samples, "ekip calismasi", "Acik iletisim ve net gorev paylasimi ekip calismasini guclendirir.");
  AppendSample(samples, "toplanti notu", "Toplanti notunda kararlar, sorumlular ve tarihler net yazilmalidir.");
  AppendSample(samples, "email yaz", "Kisa, net ve nazik bir email genelde en etkilisidir.");
  AppendSample(samples, "sunum yapacagim", "Sunumda az metin, net basliklar ve ornekler kullanin.");
  AppendSample(samples, "resume nasil olmali", "Resume'de deneyimler, beceriler ve somut sonuclar net yazilmalidir.");
  AppendSample(samples, "mulakata hazirlik", "Temel kavramlari tekrar edin ve yaptiginiz projeleri net anlatabilin.");
  AppendSample(samples, "is bulmak zor", "Duzenli basvuru, portfolyo gelistirme ve pratik mulakat fark yaratir.");
  AppendSample(samples, "ogrenmeye nereden baslamaliyim", "Ilgi duydugunuz alanda kucuk bir proje secip uygulayarak baslayin.");
  AppendSample(samples, "bugun ne yapalim", "Bugun modeli egitip uygulamada test etmek iyi bir plan olabilir.");
  AppendSample(samples, "son soz", "Son soz olarak, kucuk adimlarla ilerlemek en saglam yoldur.");

  return samples;
}

std::string BuildTokenizerCorpus(const std::vector<DialogueSample>& samples) {
  std::ostringstream stream;
  for (const DialogueSample& sample : samples) {
    if (!sample.system.empty()) {
      stream << sample.system << ' ';
    }
    stream << sample.prompt << ' ' << sample.response << ' ';
  }
  return stream.str();
}

std::vector<std::vector<TokenId>> BuildTrainingBatches(const BpeTokenizer& tokenizer,
                                                       const std::vector<DialogueSample>& samples,
                                                       const std::size_t maxSeqLen) {
  std::vector<std::vector<TokenId>> batches;
  batches.reserve(samples.size());
  for (const DialogueSample& sample : samples) {
    std::vector<TokenId> tokens =
        tokenizer.EncodeWithSpecialTokens(BuildTrainingSequence(sample), true, false);
    if (tokens.size() > maxSeqLen) {
      tokens.resize(maxSeqLen);
    }
    if (tokens.size() < 2) {
      continue;
    }
    batches.push_back(std::move(tokens));
  }
  return batches;
}

} // namespace llm::training
