#include "llm/training/loss.hpp"

#include <cmath>
#include <limits>

namespace llm::training {

float CrossEntropyLoss(const Tensor& logits, const std::vector<TokenId>& targetTokens, Tensor& gradLogits) {
  if (logits.Rank() != 2) {
    return std::numeric_limits<float>::infinity();
  }

  const Dimension seqLen = logits.GetShape()[0];
  const Dimension vocabSize = logits.GetShape()[1];

  if (targetTokens.size() != static_cast<std::size_t>(seqLen)) {
    return std::numeric_limits<float>::infinity();
  }

  gradLogits = Tensor::Zeros(logits.GetShape());
  float totalLoss = 0.0f;

  for (Dimension row = 0; row < seqLen; ++row) {
    const TokenId target = targetTokens[static_cast<std::size_t>(row)];
    if (target >= static_cast<TokenId>(vocabSize)) {
      return std::numeric_limits<float>::infinity();
    }

    Scalar maxLogit = logits.At({static_cast<Index>(row), 0});
    for (Dimension vocab = 1; vocab < vocabSize; ++vocab) {
      maxLogit = std::max(maxLogit, logits.At({static_cast<Index>(row), static_cast<Index>(vocab)}));
    }

    Scalar sumExp = 0.0f;
    for (Dimension vocab = 0; vocab < vocabSize; ++vocab) {
      const Scalar probability =
          std::exp(logits.At({static_cast<Index>(row), static_cast<Index>(vocab)}) - maxLogit);
      gradLogits.At({static_cast<Index>(row), static_cast<Index>(vocab)}) = probability;
      sumExp += probability;
    }

    for (Dimension vocab = 0; vocab < vocabSize; ++vocab) {
      gradLogits.At({static_cast<Index>(row), static_cast<Index>(vocab)}) /= sumExp;
    }

    const Scalar targetProbability =
        gradLogits.At({static_cast<Index>(row), static_cast<Index>(target)});
    totalLoss -= std::log(std::max(targetProbability, 1e-12f));

    gradLogits.At({static_cast<Index>(row), static_cast<Index>(target)}) -= 1.0f;
    for (Dimension vocab = 0; vocab < vocabSize; ++vocab) {
      gradLogits.At({static_cast<Index>(row), static_cast<Index>(vocab)}) /= static_cast<Scalar>(seqLen);
    }
  }

  return totalLoss / static_cast<Scalar>(seqLen);
}

} // namespace llm::training
