#include "llm/training/jsonl_parser.hpp"

#include <cctype>
#include <optional>
#include <string>

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

} // namespace

std::optional<std::string> ExtractJsonStringField(const std::string& json, const std::string& key,
                                                  const std::size_t searchFrom) {
  const std::string pattern = "\"" + key + "\"";
  const std::size_t keyPos = json.find(pattern, searchFrom);
  if (keyPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t colonPos = json.find(':', keyPos + pattern.size());
  if (colonPos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t quoteStart = json.find('"', colonPos + 1);
  if (quoteStart == std::string::npos) {
    return std::nullopt;
  }

  std::string value;
  bool escaped = false;
  for (std::size_t i = quoteStart + 1; i < json.size(); ++i) {
    const char ch = json[i];
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

std::vector<std::pair<std::string, std::string>> ExtractRoleContentPairs(const std::string& json) {
  std::vector<std::pair<std::string, std::string>> pairs;
  std::size_t searchFrom = 0;
  while (searchFrom < json.size()) {
    const std::size_t roleKeyPos = json.find("\"role\"", searchFrom);
    if (roleKeyPos == std::string::npos) {
      break;
    }

    const auto role = ExtractJsonStringField(json, "role", roleKeyPos);
    if (!role.has_value()) {
      searchFrom = roleKeyPos + 6;
      continue;
    }

    const std::size_t contentKeyPos = json.find("\"content\"", roleKeyPos);
    if (contentKeyPos == std::string::npos) {
      break;
    }

    const auto content = ExtractJsonStringField(json, "content", contentKeyPos);
    if (!content.has_value()) {
      searchFrom = contentKeyPos + 9;
      continue;
    }

    pairs.emplace_back(Trim(role.value()), Trim(content.value()));
    searchFrom = contentKeyPos + 9;
  }

  return pairs;
}

Result<DialogueSample> ParseJsonlDialogueLine(const std::string& line) {
  const std::string trimmed = Trim(line);
  if (trimmed.empty() || trimmed[0] == '#') {
    return Result<DialogueSample>::Fail(ErrorCode::InvalidArgument, "empty jsonl line");
  }

  if (trimmed[0] != '{') {
    return Result<DialogueSample>::Fail(ErrorCode::InvalidArgument, "jsonl line must start with '{'");
  }

  const auto pairs = ExtractRoleContentPairs(trimmed);
  if (!pairs.empty()) {
    std::string system;
    std::string user;
    std::string assistant;
    for (const auto& [role, content] : pairs) {
      if (content.empty()) {
        continue;
      }
      if (role == "system" && system.empty()) {
        system = content;
      } else if (role == "user" && user.empty()) {
        user = content;
      } else if (role == "assistant" && user.empty() == false && assistant.empty()) {
        assistant = content;
      }
    }

    if (!user.empty() && !assistant.empty()) {
      const auto intent = ExtractJsonStringField(trimmed, "intent");
      return Result<DialogueSample>::Ok(DialogueSample{
          .intent = intent.value_or(std::string{}),
          .system = std::move(system),
          .prompt = std::move(user),
          .response = std::move(assistant),
      });
    }
  }

  const auto instruction = ExtractJsonStringField(trimmed, "instruction");
  if (!instruction.has_value()) {
    const auto prompt = ExtractJsonStringField(trimmed, "prompt");
    const auto input = ExtractJsonStringField(trimmed, "input");
    const auto response = ExtractJsonStringField(trimmed, "response");
    const auto output = ExtractJsonStringField(trimmed, "output");
    const std::string user = prompt.has_value() && !prompt->empty()
                                 ? prompt.value()
                                 : (input.has_value() ? input.value() : std::string{});
    const std::string assistant = response.has_value() && !response->empty()
                                      ? response.value()
                                      : (output.has_value() ? output.value() : std::string{});
    if (!user.empty() && !assistant.empty()) {
      const auto intent = ExtractJsonStringField(trimmed, "intent");
      return Result<DialogueSample>::Ok(DialogueSample{
          .intent = intent.value_or(std::string{}),
          .prompt = Trim(user),
          .response = Trim(assistant),
      });
    }
    return Result<DialogueSample>::Fail(ErrorCode::InvalidArgument, "jsonl line has no user/assistant messages");
  }

  const auto response = ExtractJsonStringField(trimmed, "response");
  if (!response.has_value() || instruction->empty() || response->empty()) {
    return Result<DialogueSample>::Fail(ErrorCode::InvalidArgument, "jsonl instruction/response is empty");
  }

  return Result<DialogueSample>::Ok(DialogueSample{
      .intent = ExtractJsonStringField(trimmed, "intent").value_or(std::string{}),
      .prompt = Trim(instruction.value()),
      .response = Trim(response.value()),
  });
}

} // namespace llm::training
