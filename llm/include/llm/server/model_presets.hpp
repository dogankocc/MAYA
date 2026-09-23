#pragma once

// Eski model_presets.hpp - tum fonksiyonlar model_manager.hpp'a tasindi
// Bu dosya geriye donuk uyumluluk icin burada duruyor

#include "llm/server/model_manager.hpp"

// Eski NewModelRequest = CreateModelRequest ile ayni
namespace llm::server {

using NewModelRequest = CreateModelRequest;

} // namespace llm::server
