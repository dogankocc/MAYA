#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llm/core/status.hpp"

namespace llm::storage {

struct StorageConfig {
    enum class Type {
        Local,
        GoogleDrive,
        S3
    };

    Type type = Type::Local;

    // Local storage
    std::string localPath = ".";

    // Google Drive
    std::string driveCredentialsPath;
    std::string driveRootFolderId;
    std::string driveAccessToken;
    std::string driveClientId;
    std::string driveClientSecret;
    std::string driveRefreshToken;

    // S3/AWS
    std::string s3AccessKey;
    std::string s3SecretKey;
    std::string s3Bucket;
    std::string s3Region = "us-east-1";
};

class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    [[nodiscard]] virtual Status UploadFile(
        const std::string& localPath,
        const std::string& remotePath) = 0;

    [[nodiscard]] virtual Status DownloadFile(
        const std::string& remotePath,
        const std::string& localPath) = 0;

    [[nodiscard]] virtual Status RemoveFile(
        const std::string& remotePath) = 0;

    [[nodiscard]] virtual std::optional<std::vector<std::string>> ListFiles(
        const std::string& prefix = "") = 0;

    [[nodiscard]] virtual Status UploadModelCheckpoint(
        const std::string& localCheckpointPath,
        const std::string& modelName,
        const std::string& version = "latest") = 0;

    [[nodiscard]] virtual Status DownloadModelCheckpoint(
        const std::string& modelName,
        const std::string& localOutputPath,
        const std::string& version = "latest") = 0;
};

class LocalStorageBackend final : public StorageBackend {
public:
    explicit LocalStorageBackend(std::string rootPath);

    [[nodiscard]] Status UploadFile(
        const std::string& localPath,
        const std::string& remotePath) override;

    [[nodiscard]] Status DownloadFile(
        const std::string& remotePath,
        const std::string& localPath) override;

    [[nodiscard]] Status RemoveFile(
        const std::string& remotePath) override;

    [[nodiscard]] std::optional<std::vector<std::string>> ListFiles(
        const std::string& prefix = "") override;

    [[nodiscard]] Status UploadModelCheckpoint(
        const std::string& localCheckpointPath,
        const std::string& modelName,
        const std::string& version = "latest") override;

    [[nodiscard]] Status DownloadModelCheckpoint(
        const std::string& modelName,
        const std::string& localOutputPath,
        const std::string& version = "latest") override;

private:
    std::string rootPath_;
};

class GoogleDriveStorageBackend final : public StorageBackend {
public:
    explicit GoogleDriveStorageBackend(
        std::string credentialsPath,
        std::string rootFolderId = "");

    GoogleDriveStorageBackend(
        std::string clientId,
        std::string clientSecret,
        std::string refreshToken,
        std::string rootFolderId = "");

    [[nodiscard]] Status UploadFile(
        const std::string& localPath,
        const std::string& remotePath) override;

    [[nodiscard]] Status DownloadFile(
        const std::string& remotePath,
        const std::string& localPath) override;

    [[nodiscard]] Status RemoveFile(
        const std::string& remotePath) override;

    [[nodiscard]] std::optional<std::vector<std::string>> ListFiles(
        const std::string& prefix = "") override;

    [[nodiscard]] Status UploadModelCheckpoint(
        const std::string& localCheckpointPath,
        const std::string& modelName,
        const std::string& version = "latest") override;

    [[nodiscard]] Status DownloadModelCheckpoint(
        const std::string& modelName,
        const std::string& localOutputPath,
        const std::string& version = "latest") override;

    [[nodiscard]] Status Authenticate();

    [[nodiscard]] std::string GetAccessToken() const { return accessToken_; }

private:
    std::string clientId_;
    std::string clientSecret_;
    std::string refreshToken_;
    std::string accessToken_;
    std::string rootFolderId_;
    std::string credentialsPath_;

    [[nodiscard]] Status RefreshAccessToken();
    [[nodiscard]] Status LoadCredentialsFromFile();
    [[nodiscard]] Status CreateFolder(const std::string& name, std::string& outId);
};

[[nodiscard]] std::unique_ptr<StorageBackend> CreateStorageBackend(
    const StorageConfig& config);

} // namespace llm::storage
