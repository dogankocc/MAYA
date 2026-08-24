#pragma once

#include <cstddef>
#include <cstdint>

namespace llm {

using Scalar = float;
using Index = std::int64_t;
using TokenId = std::uint32_t;
using Dimension = std::size_t;

constexpr TokenId kInvalidToken = UINT32_MAX;
constexpr Index kInvalidIndex = -1;
constexpr std::size_t kSimdAlignment = 32;

} // namespace llm
