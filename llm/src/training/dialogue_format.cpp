#include "llm/training/dialogue_format.hpp"

#include "llm/nlp/intent_classifier.hpp"

#include <algorithm>
#include <cctype>

namespace {

std::string ToLowerAscii(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool ContainsKeyword(const std::string& haystack, const std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (haystack.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool PromptMentionsUserCar(const std::string& userPrompt) {
  const std::string text = ToLowerAscii(userPrompt);
  return ContainsKeyword(text, {"aria", "araban", "araba", "arabam", "fiat", "punto", "multiAir", "otomobil"});
}

} // namespace

namespace llm::training {

std::string ResolveIntent(const DialogueSample& sample) {
  if (!sample.intent.empty() && nlp::IsKnownIntent(sample.intent)) {
    return nlp::NormalizeIntentLabel(sample.intent);
  }
  return nlp::ClassifyIntent(sample.prompt, sample.response);
}

std::string BuildTrainingSequence(const DialogueSample& sample) {
  const std::string intent = ResolveIntent(sample);
  std::string sequence = "Niyet: " + intent + ".";
  if (!sample.system.empty()) {
    sequence += " Sistem: " + sample.system;
  }
  sequence += " Kullanici: " + sample.prompt + " Asistan: " + sample.response;
  return sequence;
}

std::string BuildInferencePrompt(const std::string& intent, const std::string& userPrompt,
                                 const std::string& systemPrompt) {
  const std::string normalizedIntent =
      intent.empty() ? nlp::DefaultIntent() : nlp::NormalizeIntentLabel(intent);
  std::string sequence = "Niyet: " + normalizedIntent + ".";
  if (!systemPrompt.empty()) {
    sequence += " Sistem: " + systemPrompt;
  }
  sequence += " Kullanici: " + userPrompt + " Asistan:";
  return sequence;
}

std::string SanitizeGeneratedResponse(std::string text) {
  const std::string marker = "Asistan:";
  const std::size_t markerPos = text.find(marker);
  if (markerPos != std::string::npos) {
    text.erase(0, markerPos + marker.size());
  }

  const std::string intentMarker = "Niyet:";
  const std::size_t intentPos = text.find(intentMarker);
  if (intentPos != std::string::npos) {
    text.erase(intentPos, text.size());
  }

  while (!text.empty() && (text.front() == ' ' || text.front() == '\n' || text.front() == '\r')) {
    text.erase(text.begin());
  }
  return text;
}

bool IsLowQualityResponse(const std::string& text) {
  if (text.size() < 4) {
    return true;
  }

  if (text.find('\xE2') != std::string::npos) {
    return true;
  }

  if (text.find("::::") != std::string::npos || text.find(":::um") != std::string::npos) {
    return true;
  }

  if (text.find("leri,leri") != std::string::npos || text.find("ifif") != std::string::npos) {
    return true;
  }

  if (text.find("CMak") != std::string::npos && text.find("CMake") == std::string::npos) {
    return true;
  }

  const std::size_t colons = static_cast<std::size_t>(std::count(text.begin(), text.end(), ':'));
  if (colons >= 3 || colons * 3 > text.size()) {
    return true;
  }

  const std::size_t commas = static_cast<std::size_t>(std::count(text.begin(), text.end(), ','));
  if (commas > text.size() / 6) {
    return true;
  }

  std::size_t letterCount = 0;
  std::size_t alnumCount = 0;
  std::size_t weirdCount = 0;
  for (const unsigned char ch : text) {
    if (std::isalpha(ch) != 0) {
      letterCount += 1;
    }
    if (std::isalnum(ch) != 0) {
      alnumCount += 1;
    } else if (ch != ' ' && ch != '.' && ch != ',' && ch != '?' && ch != '!' && ch != '\'' && ch != '-') {
      weirdCount += 1;
    }
  }

  if (alnumCount < text.size() / 4) {
    return true;
  }

  std::size_t wordCount = 0;
  std::size_t run = 0;
  for (const unsigned char ch : text) {
    if (std::isalpha(ch) != 0) {
      run += 1;
    } else if (run >= 3) {
      wordCount += 1;
      run = 0;
    } else {
      run = 0;
    }
  }
  if (run >= 3) {
    wordCount += 1;
  }

  if (wordCount < 2 || letterCount < 12) {
    return true;
  }

  return weirdCount > text.size() / 5;
}

bool IsFallbackResponse(const std::string& text) {
  return text == "Merhaba! Size nasil yardimci olabilirim?" ||
         text == "Selam! Sorularinizi bekliyorum." ||
         text == "Gorusmek uzere! Iyi gunler dilerim." ||
         text == "Rica ederim, her zaman sorabilirsiniz." ||
         text == "Ben yerel bir Turkce sohbet asistaniyim. Teknik sorulariniza yardimci olabilirim." ||
         text == "Bu konuda daha fazla detay verirseniz size ozel cevap verebilirim." ||
         text == "Sorunuzu biraz daha acik yazar misiniz? Yardimci olmaya calisirim.";
}

std::string FallbackResponseForIntent(const std::string& intent, const std::string& userPrompt) {
  if (PromptMentionsUserCar(userPrompt)) {
    const std::string text = ToLowerAscii(userPrompt);
    if (ContainsKeyword(text, {"motor", "multiAir", "multi air", "105"})) {
      return "Aria'nin motoru 1.4 benzinli MultiAir 105 hp. Hava yolunu 63 mm aluminyum boru ve K&N filtresi ile iyilestirdim.";
    }
    if (ContainsKeyword(text, {"fren", "balata"})) {
      return "Aria'da baski balatasini kendim degistirdim. Disk yuzeyini kontrol edip guvenli montaj yaptim.";
    }
    if (ContainsKeyword(text, {"aks", "koruk"})) {
      return "Aria'da akslari sokup koruklerini degistirdim. Montajda tork ve contalara dikkat ettim.";
    }
    if (ContainsKeyword(text, {"hava", "emis", "intake", "filtre", "kn"})) {
      return "Hava emis tarafinda 63 mm aluminyum borular ve K&N hava filtresi kullandim.";
    }
    return "Adi Aria. 2012 model Fiat Punto. 1.4 benzinli MultiAir motoru var ve 105 hp guc uretiyor. "
           "Gunluk kullanim ve teknik modifikasyonlar icin kullandigim arac bu.";
  }

  const std::string normalized = nlp::NormalizeIntentLabel(intent);
  const std::string prompt = nlp::ClassifyIntent(userPrompt);

  if (normalized == "greeting" || prompt == "greeting") {
    return "Merhaba! Size nasil yardimci olabilirim?";
  }
  if (normalized == "farewell" || prompt == "farewell") {
    return "Gorusmek uzere! Iyi gunler dilerim.";
  }
  if (normalized == "gratitude" || prompt == "gratitude") {
    return "Rica ederim, her zaman sorabilirsiniz.";
  }
  if (normalized == "self_intro" || prompt == "self_intro") {
    return "Ben yerel bir Turkce sohbet asistaniyim. Teknik sorulariniza yardimci olabilirim.";
  }
  if (normalized == "personal_profile" || prompt == "personal_profile") {
    return "Bu konuda daha fazla detay verirseniz size ozel cevap verebilirim.";
  }

  return "Sorunuzu biraz daha acik yazar misiniz? Yardimci olmaya calisirim.";
}

} // namespace llm::training
