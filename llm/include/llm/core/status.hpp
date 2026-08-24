#pragma once

#include <string>
#include <utility>
#include <variant>

namespace llm {

enum class ErrorCode {
  Ok = 0,
  InvalidArgument,
  OutOfRange,
  IoError,
  NotImplemented,
  Internal
};

struct Error {
  ErrorCode code = ErrorCode::Ok;
  std::string message;
};

class Status {
public:
  Status() = default;

  static Status Ok() { return Status{}; }

  static Status Fail(ErrorCode code, std::string message) {
    Status status;
    status.error_.code = code;
    status.error_.message = std::move(message);
    return status;
  }

  [[nodiscard]] bool IsOk() const { return error_.code == ErrorCode::Ok; }

  [[nodiscard]] const Error& GetError() const { return error_; }

  [[nodiscard]] const std::string& Message() const { return error_.message; }

private:
  Error error_;
};

template <typename T>
class Result {
public:
  static Result Ok(T value) {
    Result result;
    result.data_ = std::move(value);
    return result;
  }

  static Result Fail(ErrorCode code, std::string message) {
    Result result;
    result.data_ = Error{code, std::move(message)};
    return result;
  }

  [[nodiscard]] bool IsOk() const { return std::holds_alternative<T>(data_); }

  [[nodiscard]] const T& Value() const { return std::get<T>(data_); }

  [[nodiscard]] T& Value() { return std::get<T>(data_); }

  [[nodiscard]] const Error& GetError() const { return std::get<Error>(data_); }

private:
  std::variant<T, Error> data_{Error{ErrorCode::Internal, "uninitialized"}};
};

} // namespace llm
