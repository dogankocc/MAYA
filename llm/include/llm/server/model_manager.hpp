#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llm/core/status.hpp"
#include "llm/model/params.hpp"

namespace llm::server {

namespace fs = std::filesystem;

// ========== Model Preset (Onceden tanimli optimize konfigurasyonlar) ==========
enum class ModelPreset {
    Custom = 0,
    Tiny = 1,       // 4 katman, 128 dim (~10M parametre)
    Small = 2,      // 12 katman, 384 dim (~100M parametre)
    Medium = 3,     // 24 katman, 768 dim (~350M parametre) - ONERILEN
    Large = 4,      // 32 katman, 1024 dim (~1.2B parametre)
    XLarge = 5      // 40 katman, 2048 dim (~7B+ parametre)
};

// ========== Model Kapasite Bilgisi ==========
struct ModelCapacity {
    std::int64_t totalParameters = 0;
    std::int64_t activeParameters = 0;
    float approxGbFp32 = 0.0f;
    float approxGbInt8 = 0.0f;
    std::string complexityLevel;
};

// ========== Preset Sabitleri ==========
constexpr std::int32_t kPresetTiny_NumLayers = 4;
constexpr std::int32_t kPresetTiny_HiddenDim = 128;
constexpr std::int32_t kPresetTiny_NumHeads = 2;
constexpr std::int32_t kPresetTiny_NumKvHeads = 2;
constexpr std::int32_t kPresetTiny_IntermediateDim = 512;
constexpr std::int32_t kPresetTiny_MaxSeqLen = 128;
constexpr std::int32_t kPresetTiny_VocabSize = 4000;
constexpr float kPresetTiny_LearningRate = 5e-4f;
constexpr std::int32_t kPresetTiny_BatchSize = 16;

constexpr std::int32_t kPresetSmall_NumLayers = 12;
constexpr std::int32_t kPresetSmall_HiddenDim = 384;
constexpr std::int32_t kPresetSmall_NumHeads = 6;
constexpr std::int32_t kPresetSmall_NumKvHeads = 3;
constexpr std::int32_t kPresetSmall_IntermediateDim = 1536;
constexpr std::int32_t kPresetSmall_MaxSeqLen = 256;
constexpr std::int32_t kPresetSmall_VocabSize = 8000;
constexpr float kPresetSmall_LearningRate = 4e-4f;
constexpr std::int32_t kPresetSmall_BatchSize = 12;

constexpr std::int32_t kPresetMedium_NumLayers = 24;
constexpr std::int32_t kPresetMedium_HiddenDim = 768;
constexpr std::int32_t kPresetMedium_NumHeads = 12;
constexpr std::int32_t kPresetMedium_NumKvHeads = 6;
constexpr std::int32_t kPresetMedium_IntermediateDim = 3072;
constexpr std::int32_t kPresetMedium_MaxSeqLen = 512;
constexpr std::int32_t kPresetMedium_VocabSize = 8000;
constexpr float kPresetMedium_LearningRate = 3e-4f;
constexpr std::int32_t kPresetMedium_BatchSize = 8;

constexpr std::int32_t kPresetLarge_NumLayers = 32;
constexpr std::int32_t kPresetLarge_HiddenDim = 1024;
constexpr std::int32_t kPresetLarge_NumHeads = 16;
constexpr std::int32_t kPresetLarge_NumKvHeads = 8;
constexpr std::int32_t kPresetLarge_IntermediateDim = 4096;
constexpr std::int32_t kPresetLarge_MaxSeqLen = 1024;
constexpr std::int32_t kPresetLarge_VocabSize = 16000;
constexpr float kPresetLarge_LearningRate = 2e-4f;
constexpr std::int32_t kPresetLarge_BatchSize = 4;

constexpr std::int32_t kPresetXLarge_NumLayers = 40;
constexpr std::int32_t kPresetXLarge_HiddenDim = 2048;
constexpr std::int32_t kPresetXLarge_NumHeads = 32;
constexpr std::int32_t kPresetXLarge_NumKvHeads = 16;
constexpr std::int32_t kPresetXLarge_IntermediateDim = 8192;
constexpr std::int32_t kPresetXLarge_MaxSeqLen = 2048;
constexpr std::int32_t kPresetXLarge_VocabSize = 32000;
constexpr float kPresetXLarge_LearningRate = 1e-4f;
constexpr std::int32_t kPresetXLarge_BatchSize = 2;

// ========== Model Egitim Durumu ==========
enum class TrainingStatus {
    None = 0,
    Pending = 1,
    Running = 2,
    Completed = 3,
    Failed = 4,
    Paused = 5
};

// ========== Basit Model Bilgisi (Liste icin) ==========
struct ModelInfoBasic {
    std::string id;
    std::string name;
    std::string version;
    std::string path;
    std::uint64_t fileSizeBytes = 0;
    bool isQuantized = false;
    std::string createdAt;

    std::int32_t numLayers = 0;
    std::int32_t hiddenDim = 0;
    std::int32_t vocabSize = 0;
};

// ========== Tam Model Bilgisi ==========
struct ModelInfo {
    std::string id;
    std::string name;
    std::string version;
    std::string path;
    std::string description;

    // Dosya bilgisi
    std::uint64_t fileSizeBytes = 0;
    bool isQuantized = false;
    std::string createdAt;
    std::string updatedAt;

    // Mimari parametreler
    std::int32_t numLayers = 0;
    std::int32_t hiddenDim = 0;
    std::int32_t numHeads = 0;
    std::int32_t numKvHeads = 0;
    std::int32_t intermediateDim = 0;
    std::int32_t maxSeqLen = 0;
    std::int32_t vocabSize = 0;

    // Preset
    ModelPreset preset = ModelPreset::Custom;
    std::string presetName;

    // Kapasite
    ModelCapacity capacity;

    // Egitim bilgisi
    TrainingStatus trainingStatus = TrainingStatus::None;
    std::int64_t currentStep = 0;
    std::int64_t targetSteps = 0;
    float currentLoss = 0.0f;
    float bestLoss = 0.0f;
    std::int64_t bestStep = 0;
    float learningRate = 0.0f;
    std::string corpusPath;
    std::string errorMessage;

    // Upload bilgisi
    bool uploadToGoogleDrive = false;
    std::string googleDriveFolderId;
};

// ========== Yeni Model Olusturma Istegi ==========
struct CreateModelRequest {
    std::string name;
    std::string version;
    std::string description;

    // Preset veya ozel parametreler
    ModelPreset preset = ModelPreset::Medium;
    std::int32_t numLayers = 0;
    std::int32_t hiddenDim = 0;
    std::int32_t numHeads = 0;
    std::int32_t numKvHeads = 0;
    std::int32_t intermediateDim = 0;
    std::int32_t maxSeqLen = 0;
    std::int32_t vocabSize = 8000;

    // Egitim parametreleri
    std::string corpusPath;
    std::int64_t maxSteps = 10000;
    float learningRate = 3e-4f;
    std::int32_t batchSize = 8;
    bool quantizeAfterTraining = true;

    // Upload
    bool uploadToGoogleDrive = false;
    std::string googleDriveFolderId;
    std::string googleDriveClientId;
    std::string googleDriveClientSecret;
    std::string googleDriveRefreshToken;
};

// ========== Model Kapasite Hesaplama ==========
inline ModelCapacity CalculateModelCapacity(
    std::int32_t numLayers,
    std::int32_t hiddenDim,
    std::int32_t numHeads,
    std::int32_t intermediateDim,
    std::int32_t vocabSize,
    std::int32_t maxSeqLen = 512) {
    (void)numHeads;
    (void)maxSeqLen;

    ModelCapacity capacity;
    const std::int64_t attentionPerLayer = 4LL * static_cast<std::int64_t>(hiddenDim) * static_cast<std::int64_t>(hiddenDim);
    const std::int64_t ffnPerLayer = 3LL * static_cast<std::int64_t>(hiddenDim) * static_cast<std::int64_t>(intermediateDim);
    const std::int64_t normPerLayer = 2LL * static_cast<std::int64_t>(hiddenDim);
    const std::int64_t layersTotal = numLayers * (attentionPerLayer + ffnPerLayer + normPerLayer);
    const std::int64_t embeddingTotal = 2LL * static_cast<std::int64_t>(vocabSize) * static_cast<std::int64_t>(hiddenDim);

    capacity.totalParameters = layersTotal + embeddingTotal;
    capacity.activeParameters = capacity.totalParameters;
    capacity.approxGbFp32 = static_cast<float>(capacity.totalParameters * 4LL) / (1024.0f * 1024.0f * 1024.0f);
    capacity.approxGbInt8 = static_cast<float>(capacity.totalParameters * 1LL) / (1024.0f * 1024.0f * 1024.0f);

    if (capacity.totalParameters < 100'000'000LL) {
        capacity.complexityLevel = "Basit (Micro)";
    } else if (capacity.totalParameters < 1'000'000'000LL) {
        capacity.complexityLevel = "Orta (1B)";
    } else if (capacity.totalParameters < 7'000'000'000LL) {
        capacity.complexityLevel = "Karmasik (7B)";
    } else {
        capacity.complexityLevel = "Cok Karmasik (13B+)";
    }
    return capacity;
}

// ========== Preset Uygulama ==========
inline void ApplyModelPreset(ModelPreset preset, CreateModelRequest& request) {
    if (preset == ModelPreset::Custom) {
        if (request.numKvHeads == 0 && request.numHeads > 0) request.numKvHeads = request.numHeads / 2;
        if (request.intermediateDim == 0 && request.hiddenDim > 0) request.intermediateDim = request.hiddenDim * 4;
        if (request.maxSeqLen == 0) request.maxSeqLen = 512;
        return;
    }
    switch (preset) {
        case ModelPreset::Tiny:
            request.numLayers = kPresetTiny_NumLayers;
            request.hiddenDim = kPresetTiny_HiddenDim;
            request.numHeads = kPresetTiny_NumHeads;
            request.numKvHeads = kPresetTiny_NumKvHeads;
            request.intermediateDim = kPresetTiny_IntermediateDim;
            request.maxSeqLen = kPresetTiny_MaxSeqLen;
            request.vocabSize = kPresetTiny_VocabSize;
            if (request.learningRate <= 0) request.learningRate = kPresetTiny_LearningRate;
            if (request.batchSize <= 0) request.batchSize = kPresetTiny_BatchSize;
            break;
        case ModelPreset::Small:
            request.numLayers = kPresetSmall_NumLayers;
            request.hiddenDim = kPresetSmall_HiddenDim;
            request.numHeads = kPresetSmall_NumHeads;
            request.numKvHeads = kPresetSmall_NumKvHeads;
            request.intermediateDim = kPresetSmall_IntermediateDim;
            request.maxSeqLen = kPresetSmall_MaxSeqLen;
            request.vocabSize = kPresetSmall_VocabSize;
            if (request.learningRate <= 0) request.learningRate = kPresetSmall_LearningRate;
            if (request.batchSize <= 0) request.batchSize = kPresetSmall_BatchSize;
            break;
        case ModelPreset::Medium:
        default:
            request.numLayers = kPresetMedium_NumLayers;
            request.hiddenDim = kPresetMedium_HiddenDim;
            request.numHeads = kPresetMedium_NumHeads;
            request.numKvHeads = kPresetMedium_NumKvHeads;
            request.intermediateDim = kPresetMedium_IntermediateDim;
            request.maxSeqLen = kPresetMedium_MaxSeqLen;
            request.vocabSize = kPresetMedium_VocabSize;
            if (request.learningRate <= 0) request.learningRate = kPresetMedium_LearningRate;
            if (request.batchSize <= 0) request.batchSize = kPresetMedium_BatchSize;
            break;
        case ModelPreset::Large:
            request.numLayers = kPresetLarge_NumLayers;
            request.hiddenDim = kPresetLarge_HiddenDim;
            request.numHeads = kPresetLarge_NumHeads;
            request.numKvHeads = kPresetLarge_NumKvHeads;
            request.intermediateDim = kPresetLarge_IntermediateDim;
            request.maxSeqLen = kPresetLarge_MaxSeqLen;
            request.vocabSize = kPresetLarge_VocabSize;
            if (request.learningRate <= 0) request.learningRate = kPresetLarge_LearningRate;
            if (request.batchSize <= 0) request.batchSize = kPresetLarge_BatchSize;
            break;
        case ModelPreset::XLarge:
            request.numLayers = kPresetXLarge_NumLayers;
            request.hiddenDim = kPresetXLarge_HiddenDim;
            request.numHeads = kPresetXLarge_NumHeads;
            request.numKvHeads = kPresetXLarge_NumKvHeads;
            request.intermediateDim = kPresetXLarge_IntermediateDim;
            request.maxSeqLen = kPresetXLarge_MaxSeqLen;
            request.vocabSize = kPresetXLarge_VocabSize;
            if (request.learningRate <= 0) request.learningRate = kPresetXLarge_LearningRate;
            if (request.batchSize <= 0) request.batchSize = kPresetXLarge_BatchSize;
            break;
    }
    request.preset = preset;
}

inline std::string GetPresetName(ModelPreset preset) {
    switch (preset) {
        case ModelPreset::Tiny:   return "Micro (Tiny) - ~10M";
        case ModelPreset::Small:  return "Kucuk (Small) - ~100M";
        case ModelPreset::Medium: return "Dengeli (Medium) - ~350M (ONERILEN)";
        case ModelPreset::Large:  return "Buyuk (Large) - ~1.2B";
        case ModelPreset::XLarge: return "Cok Buyuk (XLarge) - ~7B+";
        case ModelPreset::Custom: return "Ozel (Custom)";
        default:                   return "Dengeli (Medium)";
    }
}

inline std::string GetPresetDescription(ModelPreset preset) {
    switch (preset) {
        case ModelPreset::Tiny:   return "Cok kucuk model - hizli test. 4 katman, 128 dim.";
        case ModelPreset::Small:  return "Kucuk ama etkili. 12 katman, 384 dim.";
        case ModelPreset::Medium: return "Kalite ve hiz dengesi. ONERILEN. 24 katman, 768 dim.";
        case ModelPreset::Large:  return "Daha kaliteli. 32 katman, 1024 dim.";
        case ModelPreset::XLarge: return "En kaliteli. 40 katman, 2048 dim.";
        case ModelPreset::Custom: return "Ozel parametreler.";
        default:                   return "Dengeli (Medium)";
    }
}

inline std::string TrainingStatusToString(TrainingStatus status) {
    switch (status) {
        case TrainingStatus::None:      return "None";
        case TrainingStatus::Pending:   return "Pending";
        case TrainingStatus::Running:   return "Running";
        case TrainingStatus::Completed: return "Completed";
        case TrainingStatus::Failed:    return "Failed";
        case TrainingStatus::Paused:    return "Paused";
        default:                         return "Unknown";
    }
}

inline ModelPreset DetectPresetFromParams(std::int32_t numLayers, std::int32_t hiddenDim, std::int32_t numHeads) {
    (void)numHeads;
    if (numLayers == kPresetTiny_NumLayers && hiddenDim == kPresetTiny_HiddenDim) return ModelPreset::Tiny;
    if (numLayers == kPresetSmall_NumLayers && hiddenDim == kPresetSmall_HiddenDim) return ModelPreset::Small;
    if (numLayers == kPresetMedium_NumLayers && hiddenDim == kPresetMedium_HiddenDim) return ModelPreset::Medium;
    if (numLayers == kPresetLarge_NumLayers && hiddenDim == kPresetLarge_HiddenDim) return ModelPreset::Large;
    if (numLayers == kPresetXLarge_NumLayers && hiddenDim == kPresetXLarge_HiddenDim) return ModelPreset::XLarge;
    return ModelPreset::Custom;
}

// ========== Model Manager ==========
class ModelManager {
public:
    explicit ModelManager(std::string modelsDir);

    // Temel CRUD islemleri
    [[nodiscard]] std::vector<ModelInfoBasic> ListModels();
    [[nodiscard]] std::optional<ModelInfoBasic> GetModel(const std::string& id);
    [[nodiscard]] Status DeleteModel(const std::string& id);

    // Kayitli modeli diskten oku
    [[nodiscard]] Status RegisterModel(
        const std::string& name,
        const std::string& version,
        const std::string& path);

private:
    std::string modelsDir_;

    // Yardimci fonksiyonlar
    [[nodiscard]] std::string GenerateId();
    [[nodiscard]] std::string GetTimestamp();
    [[nodiscard]] std::uint64_t GetFileSize(const std::string& path);
    [[nodiscard]] bool IsQuantizedPath(const std::string& path);
};

} // namespace llm::server
