#include "llm/training/transformer_train.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "llm/core/config.hpp"
#include "llm/model/attention.hpp"
#include "llm/model/rms_norm.hpp"
#include "llm/model/rope.hpp"
#include "llm/tensor/ops.hpp"
#include "llm/training/autograd.hpp"
#include "llm/training/loss.hpp"

namespace llm::training {

namespace {

void AccumulateParameterGrad(ParameterList& parameters, Tensor& tensor, const Tensor& grad) {
  for (Parameter& parameter : parameters.Parameters()) {
    if (parameter.tensor == &tensor) {
      Accumulate(parameter.grad, grad);
      return;
    }
  }
}

void ApplyRopeBackward(const Tensor& gradRotated, const model::RopeCache& cache, const std::size_t numHeads,
                       const std::size_t headDim, const std::size_t positionOffset, Tensor& gradInput) {
  gradInput = Tensor::Zeros(gradRotated.GetShape());
  const Dimension seqLen = gradRotated.GetShape()[0];
  const Dimension rowWidth = gradRotated.GetShape()[1];

  for (Dimension seq = 0; seq < seqLen; ++seq) {
    for (std::size_t head = 0; head < numHeads; ++head) {
      for (std::size_t dim = 0; dim < headDim; dim += 2) {
        const Index base = static_cast<Index>(seq) * rowWidth + static_cast<Index>(head * headDim + dim);
        const Scalar g0 = gradRotated[base];
        const Scalar g1 = gradRotated[base + 1];
        const Scalar cosValue = cache.Cos(positionOffset + static_cast<std::size_t>(seq), dim / 2);
        const Scalar sinValue = cache.Sin(positionOffset + static_cast<std::size_t>(seq), dim / 2);

        gradInput[base] = g0 * cosValue + g1 * sinValue;
        gradInput[base + 1] = -g0 * sinValue + g1 * cosValue;
      }
    }
  }
}

struct AttentionTrainCache {
  Tensor attnNormed;
  Tensor queries;
  Tensor keys;
  Tensor values;
  Tensor merged;
  Tensor output;
  std::vector<std::vector<std::vector<Scalar>>> probs;
};

struct FfnTrainCache {
  Tensor input;
  Tensor gate;
  Tensor up;
  Tensor hidden;
};

struct BlockTrainCache {
  Tensor input;
  Tensor attnNormed;
  Tensor residual1;
  Tensor ffnNormed;
  AttentionTrainCache attention;
  FfnTrainCache ffn;
  Tensor output;
};

struct TrainCache {
  std::vector<TokenId> tokens;
  Tensor embedded;
  std::vector<BlockTrainCache> blocks;
  Tensor preNorm;
  Tensor normalized;
  Tensor logits;
};

Status ForwardAttention(model::MultiHeadAttention& attention, const ModelConfig& config,
                        const model::RopeCache& rope, const Tensor& input, AttentionTrainCache& cache) {
  const Dimension seqLen = input.GetShape()[0];
  const Dimension numHeads = config.numHeads;
  const Dimension numKvHeads = config.numKvHeads;
  const Dimension headDim = config.hiddenDim / numHeads;
  const Dimension headsPerKv = numHeads / numKvHeads;
  const Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(headDim));
  const Scalar maskValue = -std::numeric_limits<Scalar>::infinity();

  cache.attnNormed = input;
  const Status queryStatus = attention.QueryProjection().Forward(input, cache.queries);
  const Status keyStatus = attention.KeyProjection().Forward(input, cache.keys);
  const Status valueStatus = attention.ValueProjection().Forward(input, cache.values);
  if (!queryStatus.IsOk() || !keyStatus.IsOk() || !valueStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "attention projection failed");
  }

  model::ApplyRope(cache.queries, rope, numHeads, headDim);
  model::ApplyRope(cache.keys, rope, numKvHeads, headDim);

  cache.merged = Tensor::Zeros(Shape{seqLen, config.hiddenDim});
  cache.probs.assign(static_cast<std::size_t>(seqLen), std::vector<std::vector<Scalar>>());

  for (Dimension queryPos = 0; queryPos < seqLen; ++queryPos) {
    cache.probs[static_cast<std::size_t>(queryPos)].assign(static_cast<std::size_t>(numHeads),
                                                           std::vector<Scalar>(static_cast<std::size_t>(queryPos + 1)));

    for (Dimension head = 0; head < numHeads; ++head) {
      const Dimension kvHead = head / headsPerKv;
      auto& scores = cache.probs[static_cast<std::size_t>(queryPos)][static_cast<std::size_t>(head)];

      for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
        Scalar dot = 0.0f;
        for (Dimension dim = 0; dim < headDim; ++dim) {
          dot += cache.queries.At({static_cast<Index>(queryPos), static_cast<Index>(head * headDim + dim)}) *
                 cache.keys.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
        }
        scores[static_cast<std::size_t>(keyPos)] = dot * scale;
      }

      Scalar maxScore = scores[0];
      for (Dimension keyPos = 1; keyPos <= queryPos; ++keyPos) {
        maxScore = std::max(maxScore, scores[static_cast<std::size_t>(keyPos)]);
      }

      Scalar sumExp = 0.0f;
      for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
        scores[static_cast<std::size_t>(keyPos)] = std::exp(scores[static_cast<std::size_t>(keyPos)] - maxScore);
        sumExp += scores[static_cast<std::size_t>(keyPos)];
      }

      for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
        scores[static_cast<std::size_t>(keyPos)] /= sumExp;
      }

      for (Dimension dim = 0; dim < headDim; ++dim) {
        Scalar weighted = 0.0f;
        for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
          weighted += scores[static_cast<std::size_t>(keyPos)] *
                      cache.values.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
        }
        cache.merged.At({static_cast<Index>(queryPos), static_cast<Index>(head * headDim + dim)}) = weighted;
      }
    }
  }

  return attention.OutputProjection().Forward(cache.merged, cache.output);
}

Status BackwardAttention(model::MultiHeadAttention& attention, const ModelConfig& config,
                           const model::RopeCache& rope, ParameterList& parameters, const AttentionTrainCache& cache,
                           const Tensor& gradOutput, Tensor& gradInput) {
  const Dimension seqLen = gradOutput.GetShape()[0];
  const Dimension numHeads = config.numHeads;
  const Dimension numKvHeads = config.numKvHeads;
  const Dimension headDim = config.hiddenDim / numHeads;
  const Dimension headsPerKv = numHeads / numKvHeads;
  const Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(headDim));

  Tensor gradMerged;
  Tensor gradWeightOut;
  const Status outBackward =
      LinearBackward(cache.merged, attention.OutputProjection().Weight(), gradOutput, gradMerged, gradWeightOut);
  if (!outBackward.IsOk()) {
    return outBackward;
  }
  AccumulateParameterGrad(parameters, attention.OutputProjection().WeightMutable(), gradWeightOut);

  Tensor gradQueries = Tensor::Zeros(cache.queries.GetShape());
  Tensor gradKeys = Tensor::Zeros(cache.keys.GetShape());
  Tensor gradValues = Tensor::Zeros(cache.values.GetShape());

  for (Dimension queryPos = 0; queryPos < seqLen; ++queryPos) {
    for (Dimension head = 0; head < numHeads; ++head) {
      const Dimension kvHead = head / headsPerKv;
      const auto& probs = cache.probs[static_cast<std::size_t>(queryPos)][static_cast<std::size_t>(head)];

      std::vector<Scalar> gradScores(probs.size(), 0.0f);
      for (Dimension dim = 0; dim < headDim; ++dim) {
        const Scalar gradMergedValue =
            gradMerged.At({static_cast<Index>(queryPos), static_cast<Index>(head * headDim + dim)});
        for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
          const Scalar value =
              cache.values.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          gradScores[static_cast<std::size_t>(keyPos)] += gradMergedValue * value;
          gradValues.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)}) +=
              probs[static_cast<std::size_t>(keyPos)] * gradMergedValue;
        }
      }

      for (Dimension keyPos = 0; keyPos <= queryPos; ++keyPos) {
        Scalar dot = 0.0f;
        for (Dimension inner = 0; inner <= queryPos; ++inner) {
          dot += gradScores[static_cast<std::size_t>(inner)] * probs[static_cast<std::size_t>(inner)];
        }
        gradScores[static_cast<std::size_t>(keyPos)] =
            (gradScores[static_cast<std::size_t>(keyPos)] - dot) * probs[static_cast<std::size_t>(keyPos)] * scale;

        for (Dimension dim = 0; dim < headDim; ++dim) {
          const Scalar keyValue =
              cache.keys.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)});
          gradQueries.At({static_cast<Index>(queryPos), static_cast<Index>(head * headDim + dim)}) +=
              gradScores[static_cast<std::size_t>(keyPos)] * keyValue;
          gradKeys.At({static_cast<Index>(keyPos), static_cast<Index>(kvHead * headDim + dim)}) +=
              gradScores[static_cast<std::size_t>(keyPos)] *
              cache.queries.At({static_cast<Index>(queryPos), static_cast<Index>(head * headDim + dim)});
        }
      }
    }
  }

  Tensor gradQueriesUnrotated;
  Tensor gradKeysUnrotated;
  ApplyRopeBackward(gradQueries, rope, numHeads, headDim, 0, gradQueriesUnrotated);
  ApplyRopeBackward(gradKeys, rope, numKvHeads, headDim, 0, gradKeysUnrotated);

  Tensor gradAttnInput;
  gradInput = Tensor::Zeros(cache.attnNormed.GetShape());
  Tensor gradWeightQ;
  Tensor gradWeightK;
  Tensor gradWeightV;
  Tensor gradFromQ;
  Tensor gradFromK;
  Tensor gradFromV;

  const Status qBackward =
      LinearBackward(cache.attnNormed, attention.QueryProjection().Weight(), gradQueriesUnrotated, gradFromQ, gradWeightQ);
  const Status kBackward =
      LinearBackward(cache.attnNormed, attention.KeyProjection().Weight(), gradKeysUnrotated, gradFromK, gradWeightK);
  const Status vBackward =
      LinearBackward(cache.attnNormed, attention.ValueProjection().Weight(), gradValues, gradFromV, gradWeightV);
  if (!qBackward.IsOk() || !kBackward.IsOk() || !vBackward.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "attention linear backward failed");
  }

  Accumulate(gradInput, gradFromQ);
  Accumulate(gradInput, gradFromK);
  Accumulate(gradInput, gradFromV);
  AccumulateParameterGrad(parameters, attention.QueryProjection().WeightMutable(), gradWeightQ);
  AccumulateParameterGrad(parameters, attention.KeyProjection().WeightMutable(), gradWeightK);
  AccumulateParameterGrad(parameters, attention.ValueProjection().WeightMutable(), gradWeightV);

  return Status::Ok();
}

Status BackwardFfn(model::SwiGluFfn& ffn, ParameterList& parameters, const FfnTrainCache& cache,
                   const Tensor& gradOutput, Tensor& gradInput) {
  Tensor gradHidden;
  Tensor gradWeightDown;
  const Status downBackward =
      LinearBackward(cache.hidden, ffn.DownProjection().Weight(), gradOutput, gradHidden, gradWeightDown);
  if (!downBackward.IsOk()) {
    return downBackward;
  }
  AccumulateParameterGrad(parameters, ffn.DownProjection().WeightMutable(), gradWeightDown);

  Tensor gateActivated = cache.gate;
  tensor::Silu(gateActivated);

  Tensor gradGate;
  Tensor gradUp;
  const Status mulBackward = MulBackward(gateActivated, cache.up, gradHidden, gradGate, gradUp);
  if (!mulBackward.IsOk()) {
    return mulBackward;
  }

  Tensor gradGatePre;
  SiluBackward(cache.gate, gateActivated, gradGate, gradGatePre);

  Tensor gradFromGate;
  Tensor gradWeightGate;
  Tensor gradFromUp;
  Tensor gradWeightUp;
  const Status gateLinear =
      LinearBackward(cache.input, ffn.GateProjection().Weight(), gradGatePre, gradFromGate, gradWeightGate);
  const Status upLinear = LinearBackward(cache.input, ffn.UpProjection().Weight(), gradUp, gradFromUp, gradWeightUp);
  if (!gateLinear.IsOk() || !upLinear.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "ffn linear backward failed");
  }

  gradInput = Tensor::Zeros(cache.input.GetShape());
  Accumulate(gradInput, gradFromGate);
  Accumulate(gradInput, gradFromUp);
  AccumulateParameterGrad(parameters, ffn.GateProjection().WeightMutable(), gradWeightGate);
  AccumulateParameterGrad(parameters, ffn.UpProjection().WeightMutable(), gradWeightUp);

  return Status::Ok();
}

Status ForwardBlock(model::TransformerBlock& block, const ModelConfig& config, const model::RopeCache& rope,
                    const Tensor& input, BlockTrainCache& cache) {
  cache.input = input;

  const Status norm1Status = model::RmsNorm(input, block.AttnNormWeight(), config.normEps, cache.attnNormed);
  if (!norm1Status.IsOk()) {
    return norm1Status;
  }

  const Status attnStatus = ForwardAttention(block.AttentionModule(), config, rope, cache.attnNormed, cache.attention);
  if (!attnStatus.IsOk()) {
    return attnStatus;
  }

  const Status residual1Status = tensor::Add(input, cache.attention.output, cache.residual1);
  if (!residual1Status.IsOk()) {
    return residual1Status;
  }

  const Status norm2Status = model::RmsNorm(cache.residual1, block.FfnNormWeight(), config.normEps, cache.ffnNormed);
  if (!norm2Status.IsOk()) {
    return norm2Status;
  }

  cache.ffn.input = cache.ffnNormed;
  const Status gateStatus = block.FfnModule().GateProjection().Forward(cache.ffnNormed, cache.ffn.gate);
  const Status upStatus = block.FfnModule().UpProjection().Forward(cache.ffnNormed, cache.ffn.up);
  if (!gateStatus.IsOk() || !upStatus.IsOk()) {
    return Status::Fail(ErrorCode::Internal, "ffn projection failed");
  }

  Tensor gateActivated = cache.ffn.gate;
  tensor::Silu(gateActivated);
  const Status mulStatus = tensor::Mul(gateActivated, cache.ffn.up, cache.ffn.hidden);
  if (!mulStatus.IsOk()) {
    return mulStatus;
  }

  Tensor ffnOutput;
  const Status downStatus = block.FfnModule().DownProjection().Forward(cache.ffn.hidden, ffnOutput);
  if (!downStatus.IsOk()) {
    return downStatus;
  }

  return tensor::Add(cache.residual1, ffnOutput, cache.output);
}

Status BackwardBlock(model::TransformerBlock& block, const ModelConfig& config, const model::RopeCache& rope,
                     ParameterList& parameters, const BlockTrainCache& cache, const Tensor& gradOutput,
                     Tensor& gradInput) {
  Tensor gradResidual1 = gradOutput;
  const Tensor gradFfnOutput = gradOutput;

  Tensor gradFfnNormed;
  const Status ffnBackward = BackwardFfn(block.FfnModule(), parameters, cache.ffn, gradFfnOutput, gradFfnNormed);
  if (!ffnBackward.IsOk()) {
    return ffnBackward;
  }

  Tensor gradResidualFromFfn;
  Tensor gradWeightFfn;
  const Status norm2Backward = RmsNormBackward(cache.residual1, block.FfnNormWeight(), cache.ffnNormed, gradFfnNormed,
                                               config.normEps, gradResidualFromFfn, gradWeightFfn);
  if (!norm2Backward.IsOk()) {
    return norm2Backward;
  }
  AccumulateParameterGrad(parameters, block.FfnNormWeightMutable(), gradWeightFfn);
  Accumulate(gradResidual1, gradResidualFromFfn);

  Tensor gradAttnNormed;
  const Status attnBackward = BackwardAttention(block.AttentionModule(), config, rope, parameters, cache.attention,
                                                  gradResidual1, gradAttnNormed);
  if (!attnBackward.IsOk()) {
    return attnBackward;
  }

  Tensor gradAttnInput;
  Tensor gradWeightAttn;
  const Status norm1Backward = RmsNormBackward(cache.input, block.AttnNormWeight(), cache.attnNormed, gradAttnNormed,
                                               config.normEps, gradAttnInput, gradWeightAttn);
  if (!norm1Backward.IsOk()) {
    return norm1Backward;
  }
  AccumulateParameterGrad(parameters, block.AttnNormWeightMutable(), gradWeightAttn);

  gradInput = Tensor::Zeros(cache.input.GetShape());
  Accumulate(gradInput, gradResidual1);
  Accumulate(gradInput, gradAttnInput);

  return Status::Ok();
}

} // namespace

Status RunTrainBackward(model::TransformerModel& model, ParameterList& parameters, const std::vector<TokenId>& tokens,
                        float& loss) {
  if (tokens.size() < 2) {
    return Status::Fail(ErrorCode::InvalidArgument, "training requires at least two tokens");
  }

  const ModelConfig& config = model.GetConfig();
  if (tokens.size() > config.maxSeqLen) {
    return Status::Fail(ErrorCode::InvalidArgument, "sequence length exceeds max_seq_len");
  }

  parameters.ZeroGrad();
  TrainCache cache;
  cache.tokens = tokens;

  const Dimension seqLen = tokens.size();
  cache.embedded = Tensor::Zeros(Shape{seqLen, config.hiddenDim});
  for (Dimension position = 0; position < seqLen; ++position) {
    const TokenId token = tokens[static_cast<std::size_t>(position)];
    for (Dimension dim = 0; dim < config.hiddenDim; ++dim) {
      cache.embedded.At({static_cast<Index>(position), static_cast<Index>(dim)}) =
          model.TokenEmbedding().At({static_cast<Index>(token), static_cast<Index>(dim)});
    }
  }

  Tensor current = cache.embedded;
  cache.blocks.resize(model.NumLayers());
  for (std::size_t layer = 0; layer < model.NumLayers(); ++layer) {
    Tensor next = Tensor::Zeros(Shape{seqLen, config.hiddenDim});
    const Status blockStatus = ForwardBlock(model.Layer(layer), config, model.GetRopeCache(), current, cache.blocks[layer]);
    if (!blockStatus.IsOk()) {
      return blockStatus;
    }
    current = cache.blocks[layer].output;
  }

  cache.preNorm = current;
  const Status normStatus = model::RmsNorm(current, model.FinalNormWeight(), config.normEps, cache.normalized);
  if (!normStatus.IsOk()) {
    return normStatus;
  }

  const Tensor headTransposed = model.LmHeadWeight().Transpose2D();
  const Status logitsStatus = tensor::MatMul(cache.normalized, headTransposed, cache.logits);
  if (!logitsStatus.IsOk()) {
    return logitsStatus;
  }

  const std::vector<TokenId> targets(tokens.begin() + 1, tokens.end());
  Tensor logitsForLoss = Tensor::Zeros(Shape{seqLen - 1, config.vocabSize});
  for (Dimension row = 0; row < seqLen - 1; ++row) {
    for (Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
      logitsForLoss.At({static_cast<Index>(row), static_cast<Index>(vocab)}) =
          cache.logits.At({static_cast<Index>(row), static_cast<Index>(vocab)});
    }
  }

  Tensor gradLogits;
  loss = CrossEntropyLoss(logitsForLoss, targets, gradLogits);

  Tensor fullGradLogits = Tensor::Zeros(cache.logits.GetShape());
  for (Dimension row = 0; row < seqLen - 1; ++row) {
    for (Dimension vocab = 0; vocab < config.vocabSize; ++vocab) {
      fullGradLogits.At({static_cast<Index>(row), static_cast<Index>(vocab)}) =
          gradLogits.At({static_cast<Index>(row), static_cast<Index>(vocab)});
    }
  }

  Tensor gradNormalized;
  Tensor gradLmHead;
  const Status headBackward =
      LinearBackward(cache.normalized, model.LmHeadWeight(), fullGradLogits, gradNormalized, gradLmHead);
  if (!headBackward.IsOk()) {
    return headBackward;
  }
  AccumulateParameterGrad(parameters, model.LmHeadWeightMutable(), gradLmHead);

  Tensor gradPreNorm;
  Tensor gradFinalNorm;
  const Status finalNormBackward = RmsNormBackward(cache.preNorm, model.FinalNormWeight(), cache.normalized,
                                                 gradNormalized, config.normEps, gradPreNorm, gradFinalNorm);
  if (!finalNormBackward.IsOk()) {
    return finalNormBackward;
  }
  AccumulateParameterGrad(parameters, model.FinalNormWeightMutable(), gradFinalNorm);

  Tensor gradHidden = std::move(gradPreNorm);
  for (std::size_t layer = model.NumLayers(); layer-- > 0;) {
    Tensor gradBlockInput;
    const Status blockBackward =
        BackwardBlock(model.Layer(layer), config, model.GetRopeCache(), parameters, cache.blocks[layer], gradHidden,
                      gradBlockInput);
    if (!blockBackward.IsOk()) {
      return blockBackward;
    }
    gradHidden = std::move(gradBlockInput);
  }

  for (Parameter& parameter : parameters.Parameters()) {
    if (parameter.tensor == &model.TokenEmbeddingMutable()) {
      for (Dimension position = 0; position < seqLen; ++position) {
        const TokenId token = tokens[static_cast<std::size_t>(position)];
        for (Dimension dim = 0; dim < config.hiddenDim; ++dim) {
          parameter.grad.At({static_cast<Index>(token), static_cast<Index>(dim)}) +=
              gradHidden.At({static_cast<Index>(position), static_cast<Index>(dim)});
        }
      }
      break;
    }
  }

  return Status::Ok();
}

} // namespace llm::training
