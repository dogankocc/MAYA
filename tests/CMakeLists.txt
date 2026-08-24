include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(llm_core_tests
  core_status_test.cpp
  core_config_test.cpp
  core_memory_test.cpp
  tensor_shape_test.cpp
  tensor_tensor_test.cpp
  tensor_ops_test.cpp
  tokenizer_vocabulary_test.cpp
  tokenizer_bpe_test.cpp
  model_transformer_test.cpp
  model_checkpoint_test.cpp
  inference_engine_test.cpp
  inference_sampling_test.cpp
  batch_inference_test.cpp
  continuous_batch_test.cpp
  training_test.cpp
  jsonl_corpus_test.cpp
  cli_chat_test.cpp
  quantization_test.cpp
)

target_link_libraries(llm_core_tests
  PRIVATE
    llm_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(llm_core_tests)
