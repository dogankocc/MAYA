#include "llm/nlp/intent_classifier.hpp"
#include "llm/server/openai_chat_backend.hpp"

#include <algorithm>
#include <optional>
#include "httplib.h"
#include "server/json_util.hpp"

namespace llm::server {

namespace {

[[nodiscard]] std::optional<std::string> ExtractAssistantContent(const std::string& body) {
  const std::string marker = "\"role\":\"assistant\"";
  std::size_t pos = body.find(marker);
  if (pos == std::string::npos) {
    pos = body.find("\"role\": \"assistant\"");
  }
  if (pos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t contentKey = body.find("\"content\"", pos);
  if (contentKey == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t sliceStart = contentKey;
  const std::size_t sliceEnd = std::min(body.size(), contentKey + 4096);
  return detail::ExtractStringField(body.substr(sliceStart, sliceEnd - sliceStart), "content");
}

[[nodiscard]] std::optional<std::string> ExtractErrorMessage(const std::string& body) {
  if (const auto message = detail::ExtractStringField(body, "message")) {
    return message;
  }
  return detail::ExtractStringField(body, "error");
}

} // namespace

OpenAiChatBackend::OpenAiChatBackend(OpenAiBackendConfig config) : config_(std::move(config)) {
  messages_.push_back(Message{.role = "system", .content = config_.systemPrompt});
}

std::string OpenAiChatBackend::BuildRequestBody(const ChatRequest& request) const {
  std::string body = "{\"model\":\"" + detail::JsonEscape(config_.model) + "\",\"messages\":[";
  bool first = true;
  for (const Message& message : messages_) {
    if (!first) {
      body += ',';
    }
    first = false;
    body += "{\"role\":\"" + detail::JsonEscape(message.role) + "\",\"content\":\"" +
            detail::JsonEscape(message.content) + "\"}";
  }
  if (!first) {
    body += ',';
  }
  body += "{\"role\":\"user\",\"content\":\"" + detail::JsonEscape(request.prompt) + "\"}";
  body += "],\"stream\":false,\"max_tokens\":" + std::to_string(request.maxTokens);
  body += ",\"temperature\":" + std::to_string(request.greedy ? 0.0f : request.temperature);
  body += '}';
  return body;
}

ChatResponse OpenAiChatBackend::Complete(const ChatRequest& request) {
  ChatResponse response;

  if (request.prompt.empty()) {
    response.error = "prompt is empty";
    return response;
  }

  httplib::Headers headers = {{"Content-Type", "application/json"}};
  if (!config_.apiKey.empty()) {
    headers.emplace("Authorization", "Bearer " + config_.apiKey);
  }

  const std::string requestBody = BuildRequestBody(request);

  httplib::Client client("http://" + config_.host);
  client.set_connection_timeout(10, 0);
  client.set_read_timeout(config_.timeoutSeconds, 0);
  client.set_write_timeout(10, 0);

  const auto httpResponse = client.Post("/v1/chat/completions", headers, requestBody, "application/json");
  if (!httpResponse) {
    response.error = "OpenAI-compatible API unreachable at http://" + config_.host +
                     " (Ollama veya LM Studio calisiyor mu?)";
    return response;
  }

  if (httpResponse->status < 200 || httpResponse->status >= 300) {
    if (const auto apiError = ExtractErrorMessage(httpResponse->body)) {
      response.error = *apiError;
    } else {
      response.error = "upstream API status " + std::to_string(httpResponse->status);
    }
    return response;
  }

  const auto content = ExtractAssistantContent(httpResponse->body);
  if (!content.has_value() || content->empty()) {
    response.error = "upstream API returned empty content";
    return response;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.push_back(Message{.role = "user", .content = request.prompt});
    messages_.push_back(Message{.role = "assistant", .content = *content});
  }

  response.text = *content;
  response.intent =
      request.intent.empty() ? nlp::ClassifyIntent(request.prompt, *content) : nlp::NormalizeIntentLabel(request.intent);
  return response;
}

Status OpenAiChatBackend::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  messages_.clear();
  messages_.push_back(Message{.role = "system", .content = config_.systemPrompt});
  return Status::Ok();
}

BackendInfo OpenAiChatBackend::GetInfo() const {
  return BackendInfo{
      .backend = "openai",
      .model = config_.model,
      .stage = "ollama-compatible",
      .loaded = true,
  };
}

} // namespace llm::server
