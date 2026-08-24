#include "llm/cli/chat_session.hpp"

#include "llm/model/checkpoint/checkpoint.hpp"
#include "llm/quantization/quant_checkpoint.hpp"

namespace llm::cli {

namespace {

[[nodiscard]] bool IsQuantizedCheckpointPath(const std::string& modelPath) {
  if (modelPath.size() >= 6 && modelPath.compare(modelPath.size() - 6, 6, ".ckptq") == 0) {
    return true;
  }
  return modelPath.size() >= 5 && modelPath.compare(modelPath.size() - 5, 5, ".qckpt") == 0;
}

[[nodiscard]] Result<model::TransformerModel> LoadModel(const std::string& modelPath) {
  if (IsQuantizedCheckpointPath(modelPath)) {
    return quantization::QuantCheckpoint::LoadRuntime(modelPath);
  }
  return model::Checkpoint::Load(modelPath);
}

} // namespace

ChatSession::ChatSession() {
  sampling_.temperature = 0.8f;
  sampling_.topK = 40;
  sampling_.topP = 0.9f;
  sampling_.eosTokenId = kEosTokenId;
}

Status ChatSession::Load(const std::string& modelPath, const std::string& tokenizerPath) {
  auto model = LoadModel(modelPath);
  if (!model.IsOk()) {
    return Status::Fail(model.GetError().code, model.GetError().message);
  }

  auto tokenizer = BpeTokenizer::Load(tokenizerPath);
  if (!tokenizer.IsOk()) {
    return Status::Fail(tokenizer.GetError().code, tokenizer.GetError().message);
  }

  model_ = std::move(model.Value());
  tokenizer_ = std::move(tokenizer.Value());
  engine_.emplace(*model_);
  generator_.emplace(*engine_, sampling_);
  loaded_ = true;
  return Status::Ok();
}

Result<std::string> ChatSession::Complete(const std::string& prompt, const std::size_t maxNewTokens,
                                            std::mt19937& rng) {
  if (!loaded_ || !generator_.has_value()) {
    return Result<std::string>::Fail(ErrorCode::InvalidArgument, "chat session is not loaded");
  }

  if (prompt.empty()) {
    return Result<std::string>::Fail(ErrorCode::InvalidArgument, "prompt is empty");
  }

  const std::vector<TokenId> promptIds = tokenizer_.EncodeWithSpecialTokens(prompt, true, false);
  const auto generated = generator_->Generate(promptIds, maxNewTokens, rng);
  if (!generated.IsOk()) {
    return Result<std::string>::Fail(generated.GetError().code, generated.GetError().message);
  }

  const std::vector<TokenId>& allIds = generated.Value();
  if (allIds.size() <= promptIds.size()) {
    return Result<std::string>::Ok(std::string{});
  }

  const std::vector<TokenId> newIds(allIds.begin() + static_cast<std::ptrdiff_t>(promptIds.size()), allIds.end());
  return Result<std::string>::Ok(tokenizer_.Decode(newIds));
}

void ChatSession::ResetConversation() {
  if (engine_.has_value()) {
    engine_->Reset();
  }
}

} // namespace llm::cli
