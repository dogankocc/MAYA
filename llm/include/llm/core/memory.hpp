#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

#include "llm/core/types.hpp"

namespace llm {

template <typename T>
class AlignedBuffer {
public:
  AlignedBuffer() = default;

  explicit AlignedBuffer(std::size_t count) { Resize(count); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept { MoveFrom(std::move(other)); }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  ~AlignedBuffer() { Release(); }

  void Resize(std::size_t count) {
    if (count == count_) {
      return;
    }

    Release();
    if (count == 0) {
      return;
    }

    data_ = static_cast<T*>(::operator new[](count * sizeof(T), std::align_val_t{kSimdAlignment}));
    count_ = count;
  }

  [[nodiscard]] T* Data() { return data_; }

  [[nodiscard]] const T* Data() const { return data_; }

  [[nodiscard]] std::size_t Size() const { return count_; }

  [[nodiscard]] bool Empty() const { return count_ == 0; }

  T& operator[](std::size_t index) { return data_[index]; }

  const T& operator[](std::size_t index) const { return data_[index]; }

private:
  void Release() {
    if (data_ != nullptr) {
      ::operator delete[](data_, std::align_val_t{kSimdAlignment});
      data_ = nullptr;
      count_ = 0;
    }
  }

  void MoveFrom(AlignedBuffer&& other) noexcept {
    data_ = other.data_;
    count_ = other.count_;
    other.data_ = nullptr;
    other.count_ = 0;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

} // namespace llm
