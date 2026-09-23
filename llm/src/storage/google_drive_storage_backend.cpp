// Windows icin WinHTTP kullan (OpenSSL kurulumu GEREKMIYOR)
// Windows'ta yerlesik olarak var, hicbir sey kurmana gerek yok
#define CPPHTTPLIB_USE_WINHTTP

#include "llm/storage/storage_backend.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <httplib.h>

namespace llm::storage {

namespace fs = std::filesystem;

namespace {

constexpr const char* kGoogleOAuthHost = "oauth2.googleapis.com";
constexpr const char* kGoogleDriveHost = "www.googleapis.com";

std::string UrlEncode(const std::string& s) {
    std::ostringstream encoded;
    encoded.fill('0');
    encoded << std::hex;

    for (const char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' || c == '~') {
            encoded << c;
        } else {
            encoded << '%' << std::setw(2) << std::uppercase
                    << static_cast<int>(static_cast<unsigned char>(c));
        }
    }

    return encoded.str();
}

std::string ReadFileToString(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// JSON'dan basit string field cikar
std::string ExtractJsonString(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":\"";
    const auto pos = json.find(pattern);
    if (pos == std::string::npos) {
        return "";
    }
    const auto start = pos + pattern.size();
    const auto end = json.find('"', start);
    if (end == std::string::npos) {
        return "";
    }
    return json.substr(start, end - start);
}

} // namespace

// ========== Constructors ==========

GoogleDriveStorageBackend::GoogleDriveStorageBackend(
    std::string credentialsPath,
    std::string rootFolderId)
    : credentialsPath_(std::move(credentialsPath))
    , rootFolderId_(std::move(rootFolderId)) {
}

GoogleDriveStorageBackend::GoogleDriveStorageBackend(
    std::string clientId,
    std::string clientSecret,
    std::string refreshToken,
    std::string rootFolderId)
    : clientId_(std::move(clientId))
    , clientSecret_(std::move(clientSecret))
    , refreshToken_(std::move(refreshToken))
    , rootFolderId_(std::move(rootFolderId)) {
}

// ========== Auth ==========

Status GoogleDriveStorageBackend::Authenticate() {
    if (!clientId_.empty() && !clientSecret_.empty() && !refreshToken_.empty()) {
        return RefreshAccessToken();
    }
    if (!credentialsPath_.empty()) {
        return LoadCredentialsFromFile();
    }
    return Status::Fail(ErrorCode::InvalidArgument, "no credentials provided");
}

Status GoogleDriveStorageBackend::RefreshAccessToken() {
    httplib::Client cli(kGoogleOAuthHost, 443);
    // WinHTTP'de varsayilan olarak sertifika dogrulama var
    // Gerekirse proxy veya ayarlar burada yapilabilir

    const std::string body =
        "client_id=" + UrlEncode(clientId_) +
        "&client_secret=" + UrlEncode(clientSecret_) +
        "&refresh_token=" + UrlEncode(refreshToken_) +
        "&grant_type=refresh_token";

    const auto res = cli.Post(
        "/token",
        body,
        "application/x-www-form-urlencoded");

    if (!res || res->status != 200) {
        std::cerr << "Token refresh failed: ";
        if (res) {
            std::cerr << "status " << res->status << ": " << res->body << "\n";
        } else {
            std::cerr << "no response\n";
        }
        return Status::Fail(ErrorCode::Internal, "token refresh failed");
    }

    const std::string token = ExtractJsonString(res->body, "access_token");
    if (token.empty()) {
        std::cerr << "No access_token in response: " << res->body << "\n";
        return Status::Fail(ErrorCode::Internal, "no access_token in response");
    }

    accessToken_ = token;
    std::cout << "GoogleDrive: Token refresh successful, token starts with: "
              << token.substr(0, std::min(token.size(), std::size_t(20))) << "...\n";

    return Status::Ok();
}

Status GoogleDriveStorageBackend::LoadCredentialsFromFile() {
    const std::string content = ReadFileToString(credentialsPath_);
    if (content.empty()) {
        return Status::Fail(ErrorCode::IoError, "cannot read credentials file");
    }

    clientId_ = ExtractJsonString(content, "client_id");
    if (clientId_.empty()) clientId_ = ExtractJsonString(content, "clientId");

    clientSecret_ = ExtractJsonString(content, "client_secret");
    if (clientSecret_.empty()) clientSecret_ = ExtractJsonString(content, "clientSecret");

    refreshToken_ = ExtractJsonString(content, "refresh_token");
    if (refreshToken_.empty()) refreshToken_ = ExtractJsonString(content, "refreshToken");

    rootFolderId_ = ExtractJsonString(content, "root_folder_id");

    if (clientId_.empty() || clientSecret_.empty() || refreshToken_.empty()) {
        std::cerr << "Loaded credentials:\n";
        std::cerr << "  clientId: " << (clientId_.empty() ? "(empty)" : clientId_.substr(0, 15)) << "...\n";
        std::cerr << "  refreshToken: " << (refreshToken_.empty() ? "(empty)" : "exists") << "\n";
        return Status::Fail(ErrorCode::InvalidArgument, "missing required fields in credentials");
    }

    return RefreshAccessToken();
}

// ========== File Ops ==========

Status GoogleDriveStorageBackend::UploadFile(
    const std::string& localPath,
    const std::string& remotePath) {

    if (accessToken_.empty()) {
        Status authStatus = Authenticate();
        if (!authStatus.IsOk()) {
            return authStatus;
        }
    }

    const std::string fileContent = ReadFileToString(localPath);
    if (fileContent.empty()) {
        return Status::Fail(ErrorCode::NotFound, "cannot read file: " + localPath);
    }

    const fs::path p(localPath);
    const std::string fileName = p.filename().string();

    std::cout << "GoogleDrive: Uploading " << fileName
              << " (" << fileContent.size() << " bytes)...\n";

    // Simple upload (kucuk dosyalar icin)
    httplib::Client cli(kGoogleDriveHost, 443);
    // WinHTTP'de varsayilan olarak sertifika dogrulama var
    // Gerekirse proxy veya ayarlar burada yapilabilir

    std::string jsonMeta;
    jsonMeta = "{\"name\":\"" + fileName + "\"";
    if (!rootFolderId_.empty()) {
        jsonMeta += ",\"parents\":[\"" + rootFolderId_ + "\"]";
    }
    jsonMeta += "}";

    const httplib::Headers headers = {
        {"Authorization", "Bearer " + accessToken_},
    };

    // Simple upload: uploadType=media
    const std::string url = "/upload/drive/v3/files?uploadType=media";

    const auto res = cli.Post(
        url.c_str(),
        headers,
        fileContent,
        "application/octet-stream");

    if (!res || (res->status != 200 && res->status != 201)) {
        std::cerr << "Upload failed: ";
        if (res) {
            std::cerr << "status " << res->status << ": " << res->body << "\n";
        } else {
            std::cerr << "no response\n";
        }
        return Status::Fail(ErrorCode::Internal, "upload failed");
    }

    const std::string fileId = ExtractJsonString(res->body, "id");
    std::cout << "GoogleDrive: Upload successful! File ID: " << fileId << "\n";
    return Status::Ok();
}

Status GoogleDriveStorageBackend::DownloadFile(
    const std::string& remotePath,
    const std::string& localPath) {

    std::cerr << "GoogleDrive: DownloadFile - implement me\n";
    return Status::Fail(ErrorCode::NotImplemented, "download not implemented yet");
}

Status GoogleDriveStorageBackend::RemoveFile(
    const std::string& remotePath) {

    std::cerr << "GoogleDrive: RemoveFile - implement me\n";
    return Status::Fail(ErrorCode::NotImplemented, "delete not implemented yet");
}

std::optional<std::vector<std::string>> GoogleDriveStorageBackend::ListFiles(
    const std::string& prefix) {

    if (accessToken_.empty()) {
        Status authStatus = Authenticate();
        if (!authStatus.IsOk()) {
            return std::nullopt;
        }
    }

    httplib::Client cli(kGoogleDriveHost, 443);
    // WinHTTP'de varsayilan olarak sertifika dogrulama var
    // Gerekirse proxy veya ayarlar burada yapilabilir

    std::string query = "/drive/v3/files?fields=files(id,name)&q=";
    if (!rootFolderId_.empty()) {
        query += "'" + UrlEncode(rootFolderId_) + "'+in+parents+and+";
    }
    query += "trashed=false";

    const httplib::Headers headers = {
        {"Authorization", "Bearer " + accessToken_},
    };

    const auto res = cli.Get(query.c_str(), headers);

    if (!res || res->status != 200) {
        std::cerr << "ListFiles failed: status " << (res ? res->status : 0) << "\n";
        return std::nullopt;
    }

    std::cout << "GoogleDrive ListFiles: " << res->body << "\n";
    return std::vector<std::string>();
}

Status GoogleDriveStorageBackend::CreateFolder(
    const std::string& name,
    std::string& outId) {

    if (accessToken_.empty()) {
        Status authStatus = Authenticate();
        if (!authStatus.IsOk()) {
            return authStatus;
        }
    }

    httplib::Client cli(kGoogleDriveHost, 443);
    // WinHTTP'de varsayilan olarak sertifika dogrulama var
    // Gerekirse proxy veya ayarlar burada yapilabilir

    std::string jsonBody;
    jsonBody = "{\"name\":\"" + name + "\",";
    jsonBody += "\"mimeType\":\"application/vnd.google-apps.folder\"";
    if (!rootFolderId_.empty()) {
        jsonBody += ",\"parents\":[\"" + rootFolderId_ + "\"]";
    }
    jsonBody += "}";

    const httplib::Headers headers = {
        {"Authorization", "Bearer " + accessToken_},
    };

    const auto res = cli.Post(
        "/drive/v3/files",
        headers,
        jsonBody,
        "application/json");

    if (!res || res->status != 200) {
        std::cerr << "CreateFolder failed: status " << (res ? res->status : 0) << "\n";
        if (res) std::cerr << "Response: " << res->body << "\n";
        return Status::Fail(ErrorCode::Internal, "create folder failed");
    }

    outId = ExtractJsonString(res->body, "id");
    std::cout << "GoogleDrive: Folder created, ID: " << outId << "\n";
    return Status::Ok();
}

// ========== Model Checkpoint Ops ==========

Status GoogleDriveStorageBackend::UploadModelCheckpoint(
    const std::string& localCheckpointPath,
    const std::string& modelName,
    const std::string& version) {

    if (accessToken_.empty()) {
        Status authStatus = Authenticate();
        if (!authStatus.IsOk()) {
            return authStatus;
        }
    }

    std::cout << "GoogleDrive: UploadModelCheckpoint\n";
    std::cout << "  Model: " << modelName << "\n";
    std::cout << "  Version: " << version << "\n";
    std::cout << "  Local: " << localCheckpointPath << "\n";

    // Direkt dosyayi yukle
    return UploadFile(localCheckpointPath, modelName + "/" + version);
}

Status GoogleDriveStorageBackend::DownloadModelCheckpoint(
    const std::string& modelName,
    const std::string& localOutputPath,
    const std::string& version) {

    std::cerr << "GoogleDrive: DownloadModelCheckpoint - not implemented\n";
    return Status::Fail(ErrorCode::NotImplemented, "model download not implemented");
}

// ========== Factory ==========

std::unique_ptr<StorageBackend> CreateStorageBackend(
    const StorageConfig& config) {

    switch (config.type) {
        case StorageConfig::Type::Local:
            return std::make_unique<LocalStorageBackend>(config.localPath);

        case StorageConfig::Type::GoogleDrive: {
            auto backend = std::make_unique<GoogleDriveStorageBackend>(
                config.driveClientId,
                config.driveClientSecret,
                config.driveRefreshToken,
                config.driveRootFolderId);
            return backend;
        }

        case StorageConfig::Type::S3:
            std::cerr << "CreateStorageBackend: S3 not implemented\n";
            return nullptr;
    }

    return nullptr;
}

} // namespace llm::storage
