#include <gtest/gtest.h>

#include "llm/core/memory.hpp"

TEST(MemoryTest, AlignedBufferAllocatesElements) {
  llm::AlignedBuffer<float> buffer(128);
  EXPECT_EQ(buffer.Size(), 128U);
  EXPECT_NE(buffer.Data(), nullptr);
}

TEST(MemoryTest, AlignedBufferSupportsIndexing) {
  llm::AlignedBuffer<float> buffer(4);
  buffer[0] = 1.0f;
  buffer[3] = 4.0f;
  EXPECT_FLOAT_EQ(buffer[0], 1.0f);
  EXPECT_FLOAT_EQ(buffer[3], 4.0f);
}

TEST(MemoryTest, AlignedBufferMoveTransfersOwnership) {
  llm::AlignedBuffer<float> source(16);
  source[0] = 2.5f;

  llm::AlignedBuffer<float> destination = std::move(source);
  EXPECT_TRUE(source.Empty());
  EXPECT_EQ(destination.Size(), 16U);
  EXPECT_FLOAT_EQ(destination[0], 2.5f);
}
