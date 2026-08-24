#pragma once

#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <string>
#include <vector>

#include "llm/core/types.hpp"

namespace llm {

class Shape {
public:
  Shape() = default;

  Shape(std::initializer_list<Dimension> dims) : dims_(dims) { Recompute(); }

  explicit Shape(std::vector<Dimension> dims) : dims_(std::move(dims)) { Recompute(); }

  [[nodiscard]] Dimension Rank() const { return dims_.size(); }

  [[nodiscard]] Dimension Numel() const { return numel_; }

  [[nodiscard]] Dimension operator[](Dimension axis) const { return dims_.at(axis); }

  [[nodiscard]] const std::vector<Dimension>& Dims() const { return dims_; }

  [[nodiscard]] const std::vector<Dimension>& Strides() const { return strides_; }

  [[nodiscard]] bool operator==(const Shape& other) const { return dims_ == other.dims_; }

  [[nodiscard]] bool operator!=(const Shape& other) const { return !(*this == other); }

  [[nodiscard]] std::string ToString() const;

  static std::vector<Dimension> ComputeStrides(const std::vector<Dimension>& dims);

private:
  void Recompute();

  std::vector<Dimension> dims_;
  std::vector<Dimension> strides_;
  Dimension numel_ = 0;
};

} // namespace llm
