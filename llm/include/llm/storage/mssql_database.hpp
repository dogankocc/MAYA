#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>

#include "llm/core/status.hpp"

namespace llm::storage::mssql {

struct DbConfig {
    std::string server;   // "localhost\\SQLEXPRESS" veya "server.database.windows.net"
    std::string database; // "MAYA_MODELS"
    std::string username; // "sa" veya Azure SQL kullanici adi
    std::string password;
    bool useIntegratedSecurity = false; // Windows Authentication
    int port = 1433;
};

struct DbRow {
    std::vector<std::string> values;
};

class MssqlDatabase {
public:
    explicit MssqlDatabase(const DbConfig& config);
    ~MssqlDatabase();

    // Kopyalama engelle
    MssqlDatabase(const MssqlDatabase&) = delete;
    MssqlDatabase& operator=(const MssqlDatabase&) = delete;

    // Tasima
    MssqlDatabase(MssqlDatabase&&) noexcept;
    MssqlDatabase& operator=(MssqlDatabase&&) noexcept;

    [[nodiscard]] Status Connect();
    void Disconnect();
    [[nodiscard]] bool IsConnected() const;

    // SQL calistir (sonuc donmez - INSERT, UPDATE, DELETE, CREATE)
    [[nodiscard]] Status ExecuteNonQuery(const std::string& sql);

    // SQL calistir ve sonuclari don (SELECT)
    [[nodiscard]] std::optional<std::vector<DbRow>> ExecuteQuery(const std::string& sql);

    // Tek bir deger don (SELECT COUNT(*), SELECT SCOPE_IDENTITY())
    [[nodiscard]] std::optional<std::string> ExecuteScalar(const std::string& sql);

    // Transaction
    [[nodiscard]] Status BeginTransaction();
    [[nodiscard]] Status CommitTransaction();
    [[nodiscard]] Status RollbackTransaction();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Yardimci fonksiyonlar
std::string EscapeString(const std::string& s);

} // namespace llm::storage::mssql
