#pragma once

#include <initializer_list>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/core/types.hpp"
#include "llm/tensor/shape.hpp"

namespace llm {

class Tensor {
public:
  Tensor() = default;

  Tensor(Shape shape, std::vector<Scalar> data);

  [[nodiscard]] static Tensor Zeros(Shape shape);

  [[nodiscard]] static Tensor Ones(Shape shape);

  [[nodiscard]] static Tensor FromBuffer(Shape shape, std::vector<Scalar> data);

  [[nodiscard]] const Shape& GetShape() const { return shape_; }

  [[nodiscard]] Dimension Rank() const { return shape_.Rank(); }

  [[nodiscard]] Dimension Numel() const { return shape_.Numel(); }

  [[nodiscard]] Scalar* Data() { return data_.data(); }

  [[nodiscard]] const Scalar* Data() const { return data_.data(); }

  [[nodiscard]] Scalar& operator[](Index index) { return data_[index]; }

  [[nodiscard]] const Scalar& operator[](Index index) const { return data_[index]; }

  [[nodiscard]] Scalar& At(std::initializer_list<Index> indices);

  [[nodiscard]] const Scalar& At(std::initializer_list<Index> indices) const;

  [[nodiscard]] Status Reshape(Shape newShape);

  [[nodiscard]] Tensor Transpose2D() const;

  void Fill(Scalar value);

private:
  [[nodiscard]] Index ResolveOffset(std::initializer_list<Index> indices) const;

  Shape shape_;
  std::vector<Scalar> data_;
};

} // namespace llm
