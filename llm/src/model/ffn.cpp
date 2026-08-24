#include "llm/model/ffn.hpp"

#include "llm/tensor/ops.hpp"

namespace llm::model {

SwiGluFfn::SwiGluFfn(const ModelConfig& config)
    : config_(config),
      gate_(config.hiddenDim, config.intermediateDim),
      up_(config.hiddenDim, config.intermediateDim),
      down_(config.intermediateDim, config.hiddenDim) {}

void SwiGluFfn::ResetParameters(std::mt19937& rng) {
  gate_.ResetParameters(rng);
  up_.ResetParameters(rng);
  down_.ResetParameters(rng);
}

Status SwiGluFfn::Forward(const Tensor& input, Tensor& output) const {
  Tensor gate;
  Tensor up;
  Tensor hidden;

  const Status gateStatus = gate_.Forward(input, gate);
  const Status upStatus = up_.Forward(input, up);
  if (!gateStatus.IsOk() || !upStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "ffn projection failed");
  }

  tensor::Silu(gate);
  const Status mulStatus = tensor::Mul(gate, up, hidden);
  if (!mulStatus.IsOk()) {
    return mulStatus;
  }

  return down_.Forward(hidden, output);
}

} // namespace llm::model
