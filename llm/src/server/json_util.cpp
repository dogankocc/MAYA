#include "server/json_util.hpp"

#include <charconv>
#include <cctype>
#include <sstream>

namespace llm::server::detail {

namespace {

[[nodiscard]] std::optional<std::string> ExtractStringFieldImpl(const std::string& body, const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  const std::size_t keyPos = body.find(pattern);
  if (keyPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t colonPos = body.find(':', keyPos + pattern.size());
  if (colonPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t quoteStart = body.find('"', colonPos + 1);
  if (quoteStart == std::string::npos) {
    return std::nullopt;
  }

  std::string value;
  bool escaped = false;
  for (std::size_t i = quoteStart + 1; i < body.size(); ++i) {
    const char ch = body[i];
    if (escaped) {
      if (ch == 'n') {
        value.push_back('\n');
      } else if (ch == 't') {
        value.push_back('\t');
      } else if (ch == 'r') {
        value.push_back('\r');
      } else if (ch == '"') {
        value.push_back('"');
      } else if (ch == '\\') {
        value.push_back('\\');
      } else {
        value.push_back(ch);
      }
      escaped = false;
      continue;
    }

    if (ch == '\\') {
      escaped = true;
      continue;
    }

    if (ch == '"') {
      return value;
    }

    value.push_back(ch);
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<bool> ExtractBoolField(const std::string& body, const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  const std::size_t keyPos = body.find(pattern);
  if (keyPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t colonPos = body.find(':', keyPos + pattern.size());
  if (colonPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t valuePos = body.find_first_not_of(" \t\r\n", colonPos + 1);
  if (valuePos == std::string::npos) {
    return std::nullopt;
  }

  if (body.compare(valuePos, 4, "true") == 0) {
    return true;
  }
  if (body.compare(valuePos, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<float> ExtractNumberField(const std::string& body, const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  const std::size_t keyPos = body.find(pattern);
  if (keyPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t colonPos = body.find(':', keyPos + pattern.size());
  if (colonPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t valuePos = body.find_first_not_of(" \t\r\n", colonPos + 1);
  if (valuePos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t valueEnd = body.find_first_of(",}\r\n", valuePos);
  const std::string numberText = body.substr(valuePos, valueEnd - valuePos);
  try {
    return std::stof(numberText);
  } catch (...) {
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<std::size_t> ExtractSizeField(const std::string& body, const std::string& key) {
  const auto value = ExtractNumberField(body, key);
  if (!value.has_value() || value.value() < 0.0f) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(value.value());
}

} // namespace

std::optional<std::string> ExtractStringField(const std::string& body, const std::string& key) {
  return ExtractStringFieldImpl(body, key);
}

std::optional<ChatRequest> ParseChatRequest(const std::string& body) {
  const auto prompt = ExtractStringFieldImpl(body, "prompt");
  if (!prompt.has_value()) {
    return std::nullopt;
  }

  ChatRequest request;
  request.prompt = prompt.value();

  if (const auto maxTokens = ExtractSizeField(body, "max_tokens")) {
    request.maxTokens = maxTokens.value();
  }
  if (const auto greedy = ExtractBoolField(body, "greedy")) {
    request.greedy = greedy.value();
  }
  if (const auto temperature = ExtractNumberField(body, "temperature")) {
    request.temperature = temperature.value();
  }
  if (const auto seed = ExtractSizeField(body, "seed")) {
    request.seed = static_cast<std::uint32_t>(seed.value());
  }
  if (const auto intent = ExtractStringFieldImpl(body, "intent")) {
    request.intent = intent.value();
  }

  return request;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream stream;
  for (const char ch : value) {
    switch (ch) {
    case '"':
      stream << "\\\"";
      break;
    case '\\':
      stream << "\\\\";
      break;
    case '\n':
      stream << "\\n";
      break;
    case '\r':
      stream << "\\r";
      break;
    case '\t':
      stream << "\\t";
      break;
    default:
      stream << ch;
      break;
    }
  }
  return stream.str();
}

std::string BuildChatResponseJson(const ChatResponse& response) {
  const std::string intentField =
      response.intent.empty() ? "null" : std::string("\"") + JsonEscape(response.intent) + "\"";
  if (response.error.empty()) {
    return std::string("{\"text\":\"") + JsonEscape(response.text) + "\",\"intent\":" + intentField +
           ",\"error\":null}";
  }
  return std::string("{\"text\":\"\",\"intent\":") + intentField + ",\"error\":\"" + JsonEscape(response.error) +
         "\"}";
}

std::string BuildHealthJson(const BackendInfo& info, const AutoLearnStatus& autoLearn) {
  std::string json = std::string("{\"status\":\"ok\",\"loaded\":") + (info.loaded ? "true" : "false") + ",\"stage\":\"" +
                     JsonEscape(info.stage) + "\",\"backend\":\"" + JsonEscape(info.backend) + "\",\"model\":\"" +
                     JsonEscape(info.model) + "\"";
  if (autoLearn.enabled) {
    json += ",\"auto_learn\":{\"enabled\":true,\"training\":";
    json += autoLearn.training ? "true" : "false";
    json += ",\"pending_samples\":" + std::to_string(autoLearn.pendingSamples);
    json += ",\"total_saved\":" + std::to_string(autoLearn.totalSaved);
    json += ",\"last_train_unix\":" + std::to_string(autoLearn.lastTrainUnix);
  if (!autoLearn.lastError.empty()) {
      json += ",\"last_error\":\"" + JsonEscape(autoLearn.lastError) + "\"";
    }
    json += "}";
  }
  json += "}";
  return json;
}

std::string BuildConfigJson(const ServerConfig& config) {
  return std::string("{\"backend\":\"") + JsonEscape(config.backend) + "\",\"model\":\"" +
         JsonEscape(config.model) + "\",\"local_available\":" + (config.localAvailable ? "true" : "false") +
         ",\"openai_available\":" + (config.openAiAvailable ? "true" : "false") + ",\"openai_model\":\"" +
         JsonEscape(config.openAiModel) + "\"}";
}

std::optional<ServerConfig> ParseConfigRequest(const std::string& body) {
  ServerConfig config;
  if (const auto backend = ExtractStringFieldImpl(body, "backend")) {
    config.backend = *backend;
  } else {
    return std::nullopt;
  }
  if (const auto model = ExtractStringFieldImpl(body, "openai_model")) {
    config.openAiModel = *model;
  }
  return config;
}

std::string BuildOkJson() {
  return "{\"ok\":true}";
}

} // namespace llm::server::detail
