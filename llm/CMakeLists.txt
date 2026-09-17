add_library(llm_core STATIC
  src/core/logger.cpp
  src/core/config.cpp
  src/tensor/shape.cpp
  src/tensor/tensor.cpp
  src/tensor/matmul.cpp
  src/tensor/ops.cpp
  src/tokenizer/vocabulary.cpp
  src/tokenizer/bpe_tokenizer.cpp
  src/model/params.cpp
  src/model/linear.cpp
  src/model/rms_norm.cpp
  src/model/rope.cpp
  src/model/attention.cpp
  src/model/ffn.cpp
  src/model/transformer_block.cpp
  src/model/transformer.cpp
  src/model/checkpoint/checkpoint.cpp
  src/inference/kv_cache.cpp
  src/inference/inference_engine.cpp
  src/inference/batch_kv_cache.cpp
  src/inference/batch_inference_engine.cpp
  src/inference/continuous_batch_engine.cpp
  src/inference/sampling/sampler.cpp
  src/inference/sampling/generator.cpp
  src/inference/sampling/batch_generator.cpp
  src/cli/chat_session.cpp
  src/cli/chat_cli.cpp
  src/server/json_util.cpp
  src/server/chat_http_server.cpp
  src/server/local_chat_backend.cpp
  src/server/openai_chat_backend.cpp
  src/server/hybrid_chat_backend.cpp
  src/quantization/quantize.cpp
  src/quantization/quant_checkpoint.cpp
  src/training/parameter.cpp
  src/training/loss.cpp
  src/training/autograd.cpp
  src/training/adamw.cpp
  src/training/transformer_train.cpp
  src/training/trainer.cpp
  src/nlp/intent_classifier.cpp
  src/training/dialogue_format.cpp
  src/training/chat_corpus.cpp
  src/training/jsonl_parser.cpp
  src/training/jsonl_writer.cpp
  src/training/fine_tune.cpp
  src/server/auto_learn_service.cpp
  src/agent/tool.cpp
  src/agent/local_agent_backend.cpp
  src/cli/agent_cli.cpp
)

target_include_directories(llm_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(llm_core PUBLIC cxx_std_20)

include(FetchContent)
FetchContent_Declare(
  cpp_httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(cpp_httplib)

target_link_libraries(llm_core PUBLIC httplib::httplib)

if(MSVC)
  target_compile_options(llm_core PRIVATE /arch:AVX2)
else()
  target_compile_options(llm_core PRIVATE -mavx2 -mfma)
endif()
