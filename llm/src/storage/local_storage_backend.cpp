#include "llm/storage/storage_backend.hpp"

#include <filesystem>
#include <fstream>

namespace llm::storage {

namespace fs = std::filesystem;

LocalStorageBackend::LocalStorageBackend(std::string rootPath)
    : rootPath_(std::move(rootPath)) {
    fs::create_directories(rootPath_);
}

Status LocalStorageBackend::UploadFile(
    const std::string& localPath,
    const std::string& remotePath) {
    const fs::path src = localPath;
    const fs::path dst = fs::path(rootPath_) / remotePath;

    if (!fs::exists(src)) {
        return Status::Fail(ErrorCode::NotFound, "source file not found: " + localPath);
    }

    fs::create_directories(dst.parent_path());

    std::error_code ec;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        return Status::Fail(ErrorCode::IoError, "failed to copy: " + ec.message());
    }

    return Status::Ok();
}

Status LocalStorageBackend::DownloadFile(
    const std::string& remotePath,
    const std::string& localPath) {
    const fs::path src = fs::path(rootPath_) / remotePath;
    const fs::path dst = localPath;

    if (!fs::exists(src)) {
        return Status::Fail(ErrorCode::NotFound, "remote file not found: " + remotePath);
    }

    fs::create_directories(dst.parent_path());

    std::error_code ec;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        return Status::Fail(ErrorCode::IoError, "failed to copy: " + ec.message());
    }

    return Status::Ok();
}

Status LocalStorageBackend::RemoveFile(
    const std::string& remotePath) {
    const fs::path p = fs::path(rootPath_) / remotePath;

    if (!fs::exists(p)) {
        return Status::Ok();
    }

    std::error_code ec;
    fs::remove(p, ec);
    if (ec) {
        return Status::Fail(ErrorCode::IoError, "failed to delete: " + ec.message());
    }

    return Status::Ok();
}

std::optional<std::vector<std::string>> LocalStorageBackend::ListFiles(
    const std::string& prefix) {
    std::vector<std::string> files;

    const fs::path searchPath = fs::path(rootPath_) / prefix;

    if (!fs::exists(searchPath)) {
        return files;
    }

    try {
        for (const auto& entry : fs::recursive_directory_iterator(searchPath)) {
            if (fs::is_regular_file(entry)) {
                const auto relPath = fs::relative(entry.path(), rootPath_);
                files.push_back(relPath.generic_string());
            }
        }
    } catch (const std::exception& e) {
        return std::nullopt;
    }

    return files;
}

Status LocalStorageBackend::UploadModelCheckpoint(
    const std::string& localCheckpointPath,
    const std::string& modelName,
    const std::string& version) {
    const std::string remoteDir = modelName + "/" + version;
    const std::string checkpointName = fs::path(localCheckpointPath).filename().string();
    const std::string remotePath = remoteDir + "/" + checkpointName;

    return UploadFile(localCheckpointPath, remotePath);
}

Status LocalStorageBackend::DownloadModelCheckpoint(
    const std::string& modelName,
    const std::string& localOutputPath,
    const std::string& version) {
    const std::string remoteDir = modelName + "/" + version;
    const std::string checkpointName = fs::path(localOutputPath).filename().string();
    const std::string remotePath = remoteDir + "/" + checkpointName;

    return DownloadFile(remotePath, localOutputPath);
}

} // namespace llm::storage
