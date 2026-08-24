#include "matmul_internal.hpp"

#include <algorithm>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define LLM_HAS_AVX2 1
#endif

namespace llm::tensor::detail {

void MatMulScalar(const Scalar* a, const Scalar* b, Scalar* c, Dimension m, Dimension k, Dimension n) {
  std::fill(c, c + static_cast<std::size_t>(m) * n, 0.0f);

  for (Dimension row = 0; row < m; ++row) {
    const Scalar* aRow = a + static_cast<std::size_t>(row) * k;
    Scalar* cRow = c + static_cast<std::size_t>(row) * n;

    for (Dimension col = 0; col < n; ++col) {
      Scalar sum = 0.0f;
      for (Dimension depth = 0; depth < k; ++depth) {
        sum += aRow[depth] * b[static_cast<std::size_t>(depth) * n + col];
      }
      cRow[col] = sum;
    }
  }
}

#if defined(LLM_HAS_AVX2)

void MatMulAvx2(const Scalar* a, const Scalar* b, Scalar* c, Dimension m, Dimension k, Dimension n) {
  std::fill(c, c + static_cast<std::size_t>(m) * n, 0.0f);

  for (Dimension row = 0; row < m; ++row) {
    const Scalar* aRow = a + static_cast<std::size_t>(row) * k;
    Scalar* cRow = c + static_cast<std::size_t>(row) * n;

    Dimension col = 0;
    for (; col + 8 <= n; col += 8) {
      __m256 sum = _mm256_setzero_ps();

      for (Dimension depth = 0; depth < k; ++depth) {
        const __m256 aVal = _mm256_set1_ps(aRow[depth]);
        const __m256 bVal = _mm256_loadu_ps(b + static_cast<std::size_t>(depth) * n + col);
        sum = _mm256_fmadd_ps(aVal, bVal, sum);
      }

      _mm256_storeu_ps(cRow + col, sum);
    }

    for (; col < n; ++col) {
      Scalar sum = 0.0f;
      for (Dimension depth = 0; depth < k; ++depth) {
        sum += aRow[depth] * b[static_cast<std::size_t>(depth) * n + col];
      }
      cRow[col] = sum;
    }
  }
}

#else

void MatMulAvx2(const Scalar* a, const Scalar* b, Scalar* c, Dimension m, Dimension k, Dimension n) {
  MatMulScalar(a, b, c, m, k, n);
}

#endif

void MatMulQuantizedScalar(const Scalar* input, const std::int8_t* weight, Scalar scale, Scalar* output, Dimension m,
                           Dimension k, Dimension n) {
  std::fill(output, output + static_cast<std::size_t>(m) * n, 0.0f);

  for (Dimension row = 0; row < m; ++row) {
    const Scalar* inputRow = input + static_cast<std::size_t>(row) * k;
    Scalar* outputRow = output + static_cast<std::size_t>(row) * n;

    for (Dimension col = 0; col < n; ++col) {
      const std::int8_t* weightRow = weight + static_cast<std::size_t>(col) * k;
      Scalar sum = 0.0f;
      for (Dimension depth = 0; depth < k; ++depth) {
        sum += inputRow[depth] * static_cast<Scalar>(weightRow[depth]);
      }
      outputRow[col] = sum * scale;
    }
  }
}

#if defined(LLM_HAS_AVX2)

void MatMulQuantizedAvx2(const Scalar* input, const std::int8_t* weight, Scalar scale, Scalar* output, Dimension m,
                         Dimension k, Dimension n) {
  std::fill(output, output + static_cast<std::size_t>(m) * n, 0.0f);

  for (Dimension row = 0; row < m; ++row) {
    const Scalar* inputRow = input + static_cast<std::size_t>(row) * k;
    Scalar* outputRow = output + static_cast<std::size_t>(row) * n;

    for (Dimension col = 0; col < n; ++col) {
      const std::int8_t* weightRow = weight + static_cast<std::size_t>(col) * k;
      __m256 sum = _mm256_setzero_ps();

      Dimension depth = 0;
      for (; depth + 8 <= k; depth += 8) {
        const __m256 inputVals = _mm256_loadu_ps(inputRow + depth);
        const __m256 weightVals = _mm256_set_ps(static_cast<Scalar>(weightRow[depth + 7]),
                                                static_cast<Scalar>(weightRow[depth + 6]),
                                                static_cast<Scalar>(weightRow[depth + 5]),
                                                static_cast<Scalar>(weightRow[depth + 4]),
                                                static_cast<Scalar>(weightRow[depth + 3]),
                                                static_cast<Scalar>(weightRow[depth + 2]),
                                                static_cast<Scalar>(weightRow[depth + 1]),
                                                static_cast<Scalar>(weightRow[depth + 0]));
        sum = _mm256_fmadd_ps(inputVals, weightVals, sum);
      }

      Scalar total = 0.0f;
      alignas(32) Scalar buffer[8];
      _mm256_store_ps(buffer, sum);
      for (int index = 0; index < 8; ++index) {
        total += buffer[index];
      }

      for (; depth < k; ++depth) {
        total += inputRow[depth] * static_cast<Scalar>(weightRow[depth]);
      }

      outputRow[col] = total * scale;
    }
  }
}

#else

void MatMulQuantizedAvx2(const Scalar* input, const std::int8_t* weight, Scalar scale, Scalar* output, Dimension m,
                         Dimension k, Dimension n) {
  MatMulQuantizedScalar(input, weight, scale, output, m, k, n);
}

#endif

} // namespace llm::tensor::detail
