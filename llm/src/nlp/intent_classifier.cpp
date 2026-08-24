#include "llm/nlp/intent_classifier.hpp"

#include <algorithm>
#include <cctype>

namespace llm::nlp {

namespace {

std::string ToLowerAscii(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool ContainsAny(const std::string& haystack, const std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (haystack.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

std::string NormalizeIntentLabel(std::string value) {
  value = ToLowerAscii(std::move(value));
  std::replace(value.begin(), value.end(), '-', '_');
  std::replace(value.begin(), value.end(), ' ', '_');
  return value;
}

bool IsKnownIntent(const std::string& intent) {
  static constexpr const char* kKnown[] = {
      "greeting",      "farewell",        "gratitude",   "self_intro",      "capability",
      "technical",     "coding",          "reasoning",   "personal_profile","wellbeing",
      "advice",        "creative",        "project_help","general",
  };
  const std::string normalized = NormalizeIntentLabel(intent);
  for (const char* known : kKnown) {
    if (normalized == known) {
      return true;
    }
  }
  return false;
}

std::string DefaultIntent() {
  return "general";
}

std::string ClassifyIntent(const std::string& userPrompt, const std::string& assistantResponse) {
  const std::string text = ToLowerAscii(userPrompt + " " + assistantResponse);

  if (ContainsAny(text, {"merhaba", "selam", "gunaydin", "iyi aksam", "naber", "nasilsin"})) {
    if (ContainsAny(text, {"kimsin", "kendini", "tanit", "sen kimsin"})) {
      return "self_intro";
    }
    return "greeting";
  }

  if (ContainsAny(text, {"gorusuruz", "hosca kal", "bay bay", "gule gule"})) {
    return "farewell";
  }

  if (ContainsAny(text, {"tesekkur", "sagol", "eyvallah", "cok oldu"})) {
    return "gratitude";
  }

  if (ContainsAny(text, {"kimsin", "kendini", "tanit", "sen kimsin", "ne yapabilirsin", "yetenek"})) {
    return "self_intro";
  }

  if (ContainsAny(text, {"ne yapabilirsin", "yardimci ol", "ozellik"})) {
    return "capability";
  }

  if (ContainsAny(text, {"kod", "debug", "hata aliyorum", "compile", "cmake", "flutter", "react native", "expo"})) {
    return "coding";
  }

  if (ContainsAny(text, {"adim adim", "mantik", "hesapla", "cozumle", "kanit", "sonuc cikar", "ortalama hiz"})) {
    return "reasoning";
  }

  if (ContainsAny(text, {"python", "api", "llm", "transformer", "tokenizer", "sinir ag", "machine learning",
                         "deep learning", "algoritma", "veritaban", "docker", "kubernetes"})) {
    return "technical";
  }

  if (ContainsAny(text, {"yorgun", "uzgun", "stres", "mutlu", "sikildim", "uyuyamiyorum"})) {
    return "wellbeing";
  }

  if (ContainsAny(text, {"oner", "tavsiye", "plan yap", "kariyer", "ogrenmeliyim", "hangi dili"})) {
    return "advice";
  }

  if (ContainsAny(text, {"hikaye", "siir", "espri", "yaz"})) {
    return "creative";
  }

  if (ContainsAny(text, {"proje", "modeli nasil", "egitim", "sunucu", "mobil"})) {
    return "project_help";
  }

  if (ContainsAny(text, {"aria", "araban", "araba", "arabam", "fiat", "punto", "multiAir", "otomobil"})) {
    return "personal_profile";
  }

  if (ContainsAny(text, {"gelistirici", "c++", "mikroservis", "gomulu", "ogrenme tarzi",
                         "calisma ortami", "hedeflerin"})) {
    return "personal_profile";
  }

  return DefaultIntent();
}

} // namespace llm::nlp
