#pragma once

#include <atomic>
#include <filesystem>
#include <string>

// Process / desktop helpers used by the `launch` command (Windows-first, POSIX fallbacks).
namespace maya::launcher {

[[nodiscard]] std::filesystem::path ExecutablePath();

// Walks up from the executable (and cwd) until a directory containing the project
// markers (CMakeLists.txt + data/ + llm/) is found. Falls back to cwd.
[[nodiscard]] std::filesystem::path ResolveProjectRoot();

// Starts an independent process (survives the launcher). `newConsole` gives it its own window.
[[nodiscard]] bool SpawnDetached(const std::string& commandLine, bool newConsole);

// Opens a URL in a chromeless browser window (Edge/Chrome --app), falling back to the default browser.
[[nodiscard]] bool OpenUrlAsApp(const std::string& url);

[[nodiscard]] bool OpenWithShell(const std::filesystem::path& path);

// Polls GET `baseUrl + path` until it answers 200 or the timeout expires.
[[nodiscard]] bool WaitForHttp(const std::string& baseUrl, const std::string& path, int timeoutMs);

// Sets `flag` on Ctrl+C / console close so long jobs can checkpoint and exit cleanly.
void InstallStopHandler(std::atomic<bool>& flag);

[[nodiscard]] std::string Quote(const std::filesystem::path& path);

} // namespace maya::launcher
