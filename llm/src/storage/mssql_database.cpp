#include "llm/storage/mssql_database.hpp"

#include <sqlext.h>
#include <sql.h>
#include <cstdio>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace llm::storage::mssql {

namespace {

std::string GetOdbcErrorMessage(SQLHANDLE handle, SQLSMALLINT handleType) {
    SQLCHAR sqlState[6] = {0};
    SQLCHAR message[SQL_MAX_MESSAGE_LENGTH] = {0};
    SQLINTEGER nativeError = 0;
    SQLSMALLINT textLength = 0;

    if (SQLGetDiagRec(handleType, handle, 1, sqlState, &nativeError,
                       message, sizeof(message), &textLength) == SQL_SUCCESS) {
        std::ostringstream oss;
        oss << "SQL State: " << sqlState
            << ", Native Error: " << nativeError
            << ", Message: " << message;
        return oss.str();
    }

    return "unknown ODBC error";
}

std::string BuildConnectionString(const DbConfig& config) {
    std::ostringstream cs;

    cs << "Driver={ODBC Driver 17 for SQL Server};";

    if (config.server.find(',') != std::string::npos) {
        cs << "Server=" << config.server << ";";
    } else {
        cs << "Server=" << config.server << "," << config.port << ";";
    }

    cs << "Database=" << config.database << ";";

    if (config.useIntegratedSecurity) {
        cs << "Trusted_Connection=Yes;";
    } else {
        cs << "UID=" << config.username << ";";
        cs << "PWD=" << config.password << ";";
    }

    // Encryption
    cs << "Encrypt=No;"; // Local MSSQL icin
    // Azure SQL icin: cs << "Encrypt=Yes;TrustServerCertificate=No;";

    return cs.str();
}

} // namespace

class MssqlDatabase::Impl {
public:
    SQLHENV henv = SQL_NULL_HENV;
    SQLHDBC hdbc = SQL_NULL_HDBC;
    bool inTransaction = false;
    DbConfig config;

    explicit Impl(const DbConfig& cfg) : config(cfg) {}

    ~Impl() {
        Disconnect();
    }

    Status Connect() {
        Disconnect();

        // Allocate environment
        if (SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &henv) != SQL_SUCCESS) {
            return Status::Fail(ErrorCode::Internal, "failed to allocate ODBC environment");
        }

        // Set ODBC version
        if (SQLSetEnvAttr(henv, SQL_ATTR_ODBC_VERSION,
                           reinterpret_cast<SQLPOINTER>(SQL_OV_ODBC3), 0) != SQL_SUCCESS) {
            return Status::Fail(ErrorCode::Internal, "failed to set ODBC version");
        }

        // Allocate connection
        if (SQLAllocHandle(SQL_HANDLE_DBC, henv, &hdbc) != SQL_SUCCESS) {
            return Status::Fail(ErrorCode::Internal, "failed to allocate ODBC connection");
        }

        // Set login timeout
        SQLSetConnectAttr(hdbc, SQL_LOGIN_TIMEOUT, reinterpret_cast<SQLPOINTER>(5), 0);

        // Build connection string
        const std::string connStr = BuildConnectionString(config);

        // Connect
        SQLCHAR outConnStr[1024] = {0};
        SQLSMALLINT outStrLen = 0;

        const SQLRETURN ret = SQLDriverConnect(
            hdbc,
            nullptr,
            reinterpret_cast<SQLCHAR*>(const_cast<char*>(connStr.c_str())),
            static_cast<SQLSMALLINT>(connStr.size()),
            outConnStr,
            sizeof(outConnStr),
            &outStrLen,
            SQL_DRIVER_NOPROMPT);

        if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
            const std::string error = GetOdbcErrorMessage(hdbc, SQL_HANDLE_DBC);
            Disconnect();
            return Status::Fail(ErrorCode::Internal, "connection failed: " + error);
        }

        return Status::Ok();
    }

    void Disconnect() {
        if (inTransaction) {
            RollbackTransaction();
        }

        if (hdbc != SQL_NULL_HDBC) {
            SQLDisconnect(hdbc);
            SQLFreeHandle(SQL_HANDLE_DBC, hdbc);
            hdbc = SQL_NULL_HDBC;
        }

        if (henv != SQL_NULL_HENV) {
            SQLFreeHandle(SQL_HANDLE_ENV, henv);
            henv = SQL_NULL_HENV;
        }
    }

    bool IsConnected() const {
        return hdbc != SQL_NULL_HDBC;
    }

    Status ExecuteNonQuery(const std::string& sql) {
        if (!IsConnected()) {
            return Status::Fail(ErrorCode::Internal, "not connected");
        }

        SQLHSTMT hstmt = SQL_NULL_HSTMT;
        if (SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &hstmt) != SQL_SUCCESS) {
            return Status::Fail(ErrorCode::Internal, "failed to allocate statement");
        }

        const SQLRETURN ret = SQLExecDirect(
            hstmt,
            reinterpret_cast<SQLCHAR*>(const_cast<char*>(sql.c_str())),
            static_cast<SQLINTEGER>(sql.size()));

        if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO && ret != SQL_NO_DATA) {
            const std::string error = GetOdbcErrorMessage(hstmt, SQL_HANDLE_STMT);
            SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
            return Status::Fail(ErrorCode::Internal, "execute failed: " + error);
        }

        SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
        return Status::Ok();
    }

    std::optional<std::vector<DbRow>> ExecuteQuery(const std::string& sql) {
        if (!IsConnected()) {
            return std::nullopt;
        }

        SQLHSTMT hstmt = SQL_NULL_HSTMT;
        if (SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &hstmt) != SQL_SUCCESS) {
            return std::nullopt;
        }

        const SQLRETURN ret = SQLExecDirect(
            hstmt,
            reinterpret_cast<SQLCHAR*>(const_cast<char*>(sql.c_str())),
            static_cast<SQLINTEGER>(sql.size()));

        if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
            const std::string error = GetOdbcErrorMessage(hstmt, SQL_HANDLE_STMT);
            SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
            std::cerr << "Query error: " << error << "\n";
            return std::nullopt;
        }

        // Get column count
        SQLSMALLINT numCols = 0;
        SQLNumResultCols(hstmt, &numCols);

        std::vector<DbRow> rows;

        // Fetch rows
        while (SQLFetch(hstmt) == SQL_SUCCESS) {
            DbRow row;
            row.values.resize(static_cast<std::size_t>(numCols));

            for (SQLSMALLINT col = 0; col < numCols; ++col) {
                SQLCHAR buffer[4096] = {0};
                SQLLEN indicator = 0;

                SQLGetData(
                    hstmt,
                    col + 1,
                    SQL_C_CHAR,
                    buffer,
                    sizeof(buffer),
                    &indicator);

                if (indicator == SQL_NULL_DATA) {
                    row.values[col] = "";
                } else {
                    row.values[col] = reinterpret_cast<char*>(buffer);
                }
            }

            rows.push_back(row);
        }

        SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
        return rows;
    }

    std::optional<std::string> ExecuteScalar(const std::string& sql) {
        const auto result = ExecuteQuery(sql);
        if (!result || result->empty() || result->front().values.empty()) {
            return std::nullopt;
        }
        return result->front().values[0];
    }

    Status BeginTransaction() {
        if (inTransaction) {
            return Status::Fail(ErrorCode::Internal, "already in transaction");
        }

        // Set autocommit off
        SQLSetConnectAttr(hdbc, SQL_ATTR_AUTOCOMMIT,
                          reinterpret_cast<SQLPOINTER>(SQL_AUTOCOMMIT_OFF), 0);
        inTransaction = true;
        return Status::Ok();
    }

    Status CommitTransaction() {
        if (!inTransaction) {
            return Status::Fail(ErrorCode::Internal, "not in transaction");
        }

        SQLEndTran(SQL_HANDLE_DBC, hdbc, SQL_COMMIT);
        SQLSetConnectAttr(hdbc, SQL_ATTR_AUTOCOMMIT,
                          reinterpret_cast<SQLPOINTER>(SQL_AUTOCOMMIT_ON), 0);
        inTransaction = false;
        return Status::Ok();
    }

    Status RollbackTransaction() {
        if (!inTransaction) {
            return Status::Fail(ErrorCode::Internal, "not in transaction");
        }

        SQLEndTran(SQL_HANDLE_DBC, hdbc, SQL_ROLLBACK);
        SQLSetConnectAttr(hdbc, SQL_ATTR_AUTOCOMMIT,
                          reinterpret_cast<SQLPOINTER>(SQL_AUTOCOMMIT_ON), 0);
        inTransaction = false;
        return Status::Ok();
    }
};

// ========== Public API ==========

MssqlDatabase::MssqlDatabase(const DbConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MssqlDatabase::~MssqlDatabase() = default;

MssqlDatabase::MssqlDatabase(MssqlDatabase&&) noexcept = default;
MssqlDatabase& MssqlDatabase::operator=(MssqlDatabase&&) noexcept = default;

Status MssqlDatabase::Connect() {
    return impl_->Connect();
}

void MssqlDatabase::Disconnect() {
    impl_->Disconnect();
}

bool MssqlDatabase::IsConnected() const {
    return impl_->IsConnected();
}

Status MssqlDatabase::ExecuteNonQuery(const std::string& sql) {
    return impl_->ExecuteNonQuery(sql);
}

std::optional<std::vector<DbRow>> MssqlDatabase::ExecuteQuery(const std::string& sql) {
    return impl_->ExecuteQuery(sql);
}

std::optional<std::string> MssqlDatabase::ExecuteScalar(const std::string& sql) {
    return impl_->ExecuteScalar(sql);
}

Status MssqlDatabase::BeginTransaction() {
    return impl_->BeginTransaction();
}

Status MssqlDatabase::CommitTransaction() {
    return impl_->CommitTransaction();
}

Status MssqlDatabase::RollbackTransaction() {
    return impl_->RollbackTransaction();
}

std::string EscapeString(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (const char ch : s) {
        if (ch == '\'') {
            result += "''";
        } else {
            result += ch;
        }
    }
    return result;
}

} // namespace llm::storage::mssql
