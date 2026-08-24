#include "llm/core/logger.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace llm {

Logger& Logger::Instance() {
  static Logger instance;
  return instance;
}

void Logger::SetLevel(LogLevel level) {
  std::lock_guard lock(mutex_);
  level_ = level;
}

void Logger::SetOutput(std::ostream& stream) {
  std::lock_guard lock(mutex_);
  output_ = &stream;
}

void Logger::Log(LogLevel level, std::string_view component, std::string_view message) {
  if (!ShouldLog(level)) {
    return;
  }

  std::lock_guard lock(mutex_);

  std::ostream& out = output_ != nullptr ? *output_ : std::cerr;

  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::tm localTime{};
#if defined(_WIN32)
  localtime_s(&localTime, &time);
#else
  localtime_r(&time, &localTime);
#endif

  out << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S")
      << '.' << std::setfill('0') << std::setw(3) << ms.count()
      << " [" << LevelName(level) << "] "
      << '[' << component << "] "
      << message << '\n';
}

void Logger::Trace(std::string_view component, std::string_view message) {
  Log(LogLevel::Trace, component, message);
}

void Logger::Debug(std::string_view component, std::string_view message) {
  Log(LogLevel::Debug, component, message);
}

void Logger::Info(std::string_view component, std::string_view message) {
  Log(LogLevel::Info, component, message);
}

void Logger::Warn(std::string_view component, std::string_view message) {
  Log(LogLevel::Warn, component, message);
}

void Logger::Error(std::string_view component, std::string_view message) {
  Log(LogLevel::Error, component, message);
}

bool Logger::ShouldLog(LogLevel level) const {
  return static_cast<int>(level) >= static_cast<int>(level_);
}

const char* Logger::LevelName(LogLevel level) {
  switch (level) {
    case LogLevel::Trace:
      return "TRACE";
    case LogLevel::Debug:
      return "DEBUG";
    case LogLevel::Info:
      return "INFO";
    case LogLevel::Warn:
      return "WARN";
    case LogLevel::Error:
      return "ERROR";
  }
  return "UNKNOWN";
}

} // namespace llm
