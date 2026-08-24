#pragma once

#include <cstdint>
#include <string>

#include "llm/core/config.hpp"
#include "llm/core/status.hpp"
#include "llm/model/transformer.hpp"

namespace llm::model {

constexpr char kCheckpointMagic[] = "LLMCKPT1";
constexpr std::uint32_t kCheckpointVersion = 1;

class Checkpoint {
public:
  [[nodiscard]] static Status Save(const TransformerModel& model, const std::string& path);

  [[nodiscard]] static Result<TransformerModel> Load(const std::string& path);
};

} // namespace llm::model
