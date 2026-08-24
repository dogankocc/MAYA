#pragma once

#include <mutex>
#include <ostream>
#include <string>
#include <string_view>

namespace llm {

enum class LogLevel {
  Trace = 0,
  Debug,
  Info,
  Warn,
  Error
};

class Logger {
public:
  static Logger& Instance();

  void SetLevel(LogLevel level);
  void SetOutput(std::ostream& stream);

  void Log(LogLevel level, std::string_view component, std::string_view message);

  void Trace(std::string_view component, std::string_view message);
  void Debug(std::string_view component, std::string_view message);
  void Info(std::string_view component, std::string_view message);
  void Warn(std::string_view component, std::string_view message);
  void Error(std::string_view component, std::string_view message);

private:
  Logger() = default;

  [[nodiscard]] bool ShouldLog(LogLevel level) const;
  [[nodiscard]] static const char* LevelName(LogLevel level);

  std::mutex mutex_;
  LogLevel level_ = LogLevel::Info;
  std::ostream* output_ = nullptr;
};

} // namespace llm
