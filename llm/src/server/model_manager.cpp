#include "llm/server/model_manager.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>

namespace llm::server {

namespace fs = std::filesystem;

namespace {

std::string GenerateRandomId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static const char* hex = "0123456789abcdef";

    std::string result;
    result.reserve(16);
    for (int i = 0; i < 16; ++i) {
        const auto idx = static_cast<std::size_t>(gen() % 16);
        result += hex[idx];
    }
    return result;
}

std::string GetCurrentTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto timeT = std::chrono::system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&timeT), "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

} // namespace

ModelManager::ModelManager(std::string modelsDir)
    : modelsDir_(std::move(modelsDir)) {
    fs::create_directories(modelsDir_);
}

std::string ModelManager::GenerateId() {
    return GenerateRandomId();
}

std::string ModelManager::GetTimestamp() {
    return GetCurrentTimestamp();
}

std::uint64_t ModelManager::GetFileSize(const std::string& path) {
    try {
        return static_cast<std::uint64_t>(fs::file_size(path));
    } catch (...) {
        return 0;
    }
}

bool ModelManager::IsQuantizedPath(const std::string& path) {
    return path.ends_with(".ckptq");
}

std::vector<ModelInfoBasic> ModelManager::ListModels() {
    std::vector<ModelInfoBasic> models;

    try {
        for (const auto& entry : fs::directory_iterator(modelsDir_)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const auto path = entry.path().string();
            const auto ext = entry.path().extension().string();

            if (ext != ".ckpt" && ext != ".ckptq") {
                continue;
            }

            ModelInfoBasic info;
            info.id = GenerateRandomId();
            info.name = entry.path().stem().string();
            info.version = "latest";
            info.path = path;
            info.fileSizeBytes = GetFileSize(path);
            info.isQuantized = IsQuantizedPath(path);
            info.createdAt = GetCurrentTimestamp();

            // TODO: Checkpoint'ten gercek bilgileri oku
            info.numLayers = 24;
            info.hiddenDim = 768;
            info.vocabSize = 8000;

            models.push_back(info);
        }
    } catch (...) {
        return models;
    }

    // Boyuta gore sirala (en buyuk en basta)
    std::sort(models.begin(), models.end(), [](const ModelInfoBasic& a, const ModelInfoBasic& b) {
        return a.fileSizeBytes > b.fileSizeBytes;
    });

    return models;
}

std::optional<ModelInfoBasic> ModelManager::GetModel(const std::string& id) {
    const auto all = ListModels();
    for (const auto& m : all) {
        if (m.id == id) {
            return m;
        }
    }
    return std::nullopt;
}

Status ModelManager::DeleteModel(const std::string& id) {
    const auto model = GetModel(id);
    if (!model) {
        return Status::Fail(ErrorCode::NotFound, "model not found: " + id);
    }

    try {
        fs::remove(model->path);
        return Status::Ok();
    } catch (const std::exception& e) {
        return Status::Fail(ErrorCode::IoError, e.what());
    }
}

Status ModelManager::RegisterModel(
    const std::string& name,
    const std::string& version,
    const std::string& path) {

    if (!fs::exists(path)) {
        return Status::Fail(ErrorCode::NotFound, "file not found: " + path);
    }

    try {
        const auto targetPath =
            fs::path(modelsDir_) /
            (name + "_" + version + (path.ends_with(".ckptq") ? ".ckptq" : ".ckpt"));

        fs::copy_file(path, targetPath, fs::copy_options::overwrite_existing);
        return Status::Ok();
    } catch (const std::exception& e) {
        return Status::Fail(ErrorCode::IoError, e.what());
    }
}

} // namespace llm::server
