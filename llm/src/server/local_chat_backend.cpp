#include "llm/inference/sampling/sampler.hpp"
#include "llm/nlp/intent_classifier.hpp"
#include "llm/server/local_chat_backend.hpp"
#include "llm/training/dialogue_format.hpp"

#include <algorithm>

namespace llm::server {

namespace {

constexpr const char* kUnknownTopicReply =
    "Bu konuda henuz yeterli bilgim yok. Daha acik sorarsan veya ornek bir cevap verirsen ogrenebilirim.";

} // namespace

Status LocalChatBackend::Load(const std::string& modelPath, const std::string& tokenizerPath) {
  retriever_.LoadDefaultKnowledgeBase();

  const Status status = session_.Load(modelPath, tokenizerPath);
  if (!status.IsOk()) {
    return status;
  }

  session_.Sampling().temperature = 0.2f;
  session_.Sampling().topK = 20;
  session_.Sampling().topP = 0.85f;
  session_.Sampling().greedy = true;
  session_.Sampling().minFirstTokenProbability = kMinGenerationConfidence;
  return Status::Ok();
}

ChatResponse LocalChatBackend::Complete(const ChatRequest& request) {
  ChatResponse response;

  if (!session_.IsLoaded()) {
    response.error = "local model is not loaded";
    return response;
  }

  if (request.prompt.empty()) {
    response.error = "prompt is empty";
    return response;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  const std::string intent =
      request.intent.empty() ? nlp::ClassifyIntent(request.prompt) : nlp::NormalizeIntentLabel(request.intent);
  response.intent = intent;

  if (const auto corpusMatch = retriever_.FindBestMatch(request.prompt, intent)) {
    response.text = corpusMatch->response;
    response.intent = corpusMatch->intent.empty() ? intent : corpusMatch->intent;
    response.allowAutoLearn = false;
    return response;
  }

  inference::SamplingConfig& sampling = session_.Sampling();
  const float savedTemperature = sampling.temperature;
  const bool savedGreedy = sampling.greedy;
  const float savedMinConfidence = sampling.minFirstTokenProbability;
  sampling.temperature = request.greedy ? 0.0f : std::min(request.temperature, 0.35f);
  sampling.greedy = request.greedy || request.temperature <= 0.05f;
  sampling.minFirstTokenProbability = kMinGenerationConfidence;

  const std::string modelPrompt = training::BuildInferencePrompt(intent, request.prompt);
  const auto result = session_.Complete(modelPrompt, request.maxTokens, rng_);

  sampling.temperature = savedTemperature;
  sampling.greedy = savedGreedy;
  sampling.minFirstTokenProbability = savedMinConfidence;

  if (!result.IsOk()) {
    response.text = kUnknownTopicReply;
    response.allowAutoLearn = false;
    return response;
  }

  response.text = training::SanitizeGeneratedResponse(result.Value());
  if (training::IsLowQualityResponse(response.text)) {
    response.text = kUnknownTopicReply;
    response.allowAutoLearn = false;
    return response;
  }

  response.allowAutoLearn = true;
  return response;
}

Status LocalChatBackend::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  session_.ResetConversation();
  return Status::Ok();
}

BackendInfo LocalChatBackend::GetInfo() const {
  return BackendInfo{
      .backend = "local",
      .model = "custom-transformer",
      .stage = "local-mini",
      .loaded = session_.IsLoaded(),
  };
}

} // namespace llm::server
