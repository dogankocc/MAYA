#pragma once

#include <cstdint>
#include <string>

#include "llm/core/status.hpp"
#include "llm/model/transformer.hpp"

namespace llm::quantization {

constexpr char kQuantCheckpointMagic[] = "LLMQCKPT";
constexpr std::uint32_t kQuantCheckpointVersion = 1;

class QuantCheckpoint {
public:
  [[nodiscard]] static Status Save(const model::TransformerModel& model, const std::string& path);

  [[nodiscard]] static Result<model::TransformerModel> Load(const std::string& path);

  [[nodiscard]] static Result<model::TransformerModel> LoadRuntime(const std::string& path);

  [[nodiscard]] static Status ConvertFile(const std::string& fp32Path, const std::string& quantPath);
};

} // namespace llm::quantization
