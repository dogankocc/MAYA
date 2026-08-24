#include <gtest/gtest.h>

#include "llm/nlp/corpus_retriever.hpp"
#include "llm/training/dialogue_format.hpp"
#include "llm/training/jsonl_parser.hpp"
#include "llm/training/jsonl_writer.hpp"

TEST(JsonlParserTest, ParsesMessagesWithIntent) {
  const std::string line =
      R"({"intent":"coding","messages":[{"role":"user","content":"python nedir"},{"role":"assistant","content":"Python bir programlama dilidir."}]})";

  const auto parsed = llm::training::ParseJsonlDialogueLine(line);
  ASSERT_TRUE(parsed.IsOk());
  EXPECT_EQ(parsed.Value().intent, "coding");
  EXPECT_EQ(parsed.Value().prompt, "python nedir");
  EXPECT_EQ(parsed.Value().response, "Python bir programlama dilidir.");
}

TEST(JsonlWriterTest, RoundTripAutoLearnLine) {
  const std::string line =
      llm::training::FormatDialogueJsonlLine("general", "selam", "Merhaba, nasil yardimci olabilirim?");
  const auto parsed = llm::training::ParseJsonlDialogueLine(line);
  ASSERT_TRUE(parsed.IsOk());
  EXPECT_EQ(parsed.Value().intent, "general");
  EXPECT_EQ(parsed.Value().prompt, "selam");
  EXPECT_EQ(parsed.Value().response, "Merhaba, nasil yardimci olabilirim?");
}

TEST(DialogueFormatTest, BuildTrainingSequenceIncludesIntentPrefix) {
  const llm::training::DialogueSample sample{
      .intent = "greeting",
      .prompt = "merhaba",
      .response = "Merhaba!",
  };

  const std::string sequence = llm::training::BuildTrainingSequence(sample);
  EXPECT_NE(sequence.find("Niyet: greeting."), std::string::npos);
  EXPECT_NE(sequence.find("Kullanici: merhaba"), std::string::npos);
  EXPECT_NE(sequence.find("Asistan: Merhaba!"), std::string::npos);
}

TEST(DialogueFormatTest, BuildInferencePromptEndsWithAssistantMarker) {
  const std::string prompt = llm::training::BuildInferencePrompt("technical", "api nedir");
  EXPECT_NE(prompt.find("Niyet: technical."), std::string::npos);
  EXPECT_TRUE(prompt.rfind("Asistan:") != std::string::npos);
}

TEST(DialogueFormatTest, RejectsColonGarbageResponse) {
  const std::string garbage =
      ":::::um.:::::::um.::::::::::::::::::::::ing:um.:um.::::::::::::iy::::::::";
  EXPECT_TRUE(llm::training::IsLowQualityResponse(garbage));
}

TEST(DialogueFormatTest, AcceptsNormalProfileResponse) {
  const std::string good =
      "Adi Aria. 2012 model Fiat Punto. 1.4 benzinli MultiAir motoru var ve 105 hp guc uretiyor.";
  EXPECT_FALSE(llm::training::IsLowQualityResponse(good));
}

TEST(DialogueFormatTest, AcceptsShortGreetingFallback) {
  EXPECT_FALSE(llm::training::IsLowQualityResponse("Merhaba! Size nasil yardimci olabilirim?"));
}

TEST(CorpusRetrieverTest, FindsGreetingFromBuiltinCorpus) {
  llm::nlp::CorpusRetriever retriever;
  retriever.LoadDefaultKnowledgeBase();

  const auto match = retriever.FindBestMatch("merhaba", "greeting");
  ASSERT_TRUE(match.has_value());
  EXPECT_NE(match->response.find("Merhaba"), std::string::npos);
}

TEST(CorpusRetrieverTest, FindsCarProfileAnswer) {
  llm::nlp::CorpusRetriever retriever;
  retriever.LoadDefaultKnowledgeBase();

  const auto match = retriever.FindBestMatch("Araba nedir", "personal_profile");
  ASSERT_TRUE(match.has_value());
  EXPECT_NE(match->response.find("Aria"), std::string::npos);
}

TEST(CorpusRetrieverTest, RejectsUnknownTopic) {
  llm::nlp::CorpusRetriever retriever;
  retriever.LoadDefaultKnowledgeBase();

  const auto match = retriever.FindBestMatch("aglamak gozlerinden yas gelmesi demektir", "general");
  EXPECT_FALSE(match.has_value());
}
