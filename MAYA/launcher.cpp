#include "launcher.hpp"

#include <chrono>
#include <csignal>
#include <thread>

#include "httplib.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <shellapi.h>
#else
#include <cstdlib>
#include <unistd.h>
#endif

namespace maya::launcher {

namespace fs = std::filesystem;

namespace {

std::atomic<bool>* g_stopFlag = nullptr;

#ifdef _WIN32
BOOL WINAPI ConsoleHandler(const DWORD /*signal*/) {
  if (g_stopFlag != nullptr) {
    g_stopFlag->store(true);
  }
  // Give the training loop a moment to flush its checkpoint before Windows kills us.
  std::this_thread::sleep_for(std::chrono::seconds(4));
  return TRUE;
}

[[nodiscard]] std::wstring Widen(const std::string& text) {
  if (text.empty()) {
    return {};
  }
  const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), wide.data(), size);
  return wide;
}

[[nodiscard]] fs::path FindBrowser() {
  for (const char* candidate : {R"(C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe)",
                                R"(C:\Program Files\Microsoft\Edge\Application\msedge.exe)",
                                R"(C:\Program Files\Google\Chrome\Application\chrome.exe)",
                                R"(C:\Program Files (x86)\Google\Chrome\Application\chrome.exe)"}) {
    if (fs::exists(candidate)) {
      return candidate;
    }
  }
  return {};
}
#else
void PosixHandler(int) {
  if (g_stopFlag != nullptr) {
    g_stopFlag->store(true);
  }
}
#endif

[[nodiscard]] bool LooksLikeProjectRoot(const fs::path& dir) {
  return fs::exists(dir / "CMakeLists.txt") && fs::is_directory(dir / "data") && fs::is_directory(dir / "llm");
}

} // namespace

fs::path ExecutablePath() {
#ifdef _WIN32
  wchar_t buffer[MAX_PATH];
  const DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
  return fs::path(std::wstring(buffer, length));
#else
  std::error_code errorCode;
  return fs::read_symlink("/proc/self/exe", errorCode);
#endif
}

fs::path ResolveProjectRoot() {
  std::error_code errorCode;
  const fs::path cwd = fs::current_path(errorCode);
  if (LooksLikeProjectRoot(cwd)) {
    return cwd;
  }

  fs::path dir = ExecutablePath().parent_path();
  for (int depth = 0; depth < 8 && !dir.empty(); ++depth) {
    if (LooksLikeProjectRoot(dir)) {
      return dir;
    }
    if (dir == dir.parent_path()) {
      break;
    }
    dir = dir.parent_path();
  }
  return cwd;
}

std::string Quote(const fs::path& path) {
  return "\"" + path.string() + "\"";
}

bool SpawnDetached(const std::string& commandLine, const bool newConsole) {
#ifdef _WIN32
  std::wstring mutableCommand = Widen(commandLine);
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  const DWORD flags = newConsole ? CREATE_NEW_CONSOLE : DETACHED_PROCESS;
  const BOOL ok = CreateProcessW(nullptr, mutableCommand.data(), nullptr, nullptr, FALSE, flags, nullptr, nullptr,
                                 &startup, &process);
  if (!ok) {
    return false;
  }
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  return true;
#else
  (void)newConsole;
  return std::system((commandLine + " &").c_str()) == 0;
#endif
}

bool OpenWithShell(const fs::path& path) {
#ifdef _WIN32
  const HINSTANCE result = ShellExecuteW(nullptr, L"open", path.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
  return reinterpret_cast<INT_PTR>(result) > 32;
#else
  return std::system(("xdg-open " + Quote(path) + " &").c_str()) == 0;
#endif
}

bool OpenUrlAsApp(const std::string& url) {
#ifdef _WIN32
  const fs::path browser = FindBrowser();
  if (!browser.empty() &&
      SpawnDetached(Quote(browser) + " --app=" + url + " --window-size=1280,900 --new-window", false)) {
    return true;
  }
#endif
  return OpenWithShell(url);
}

bool WaitForHttp(const std::string& baseUrl, const std::string& path, const int timeoutMs) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  do {
    httplib::Client client(baseUrl);
    client.set_connection_timeout(0, 500'000);
    client.set_read_timeout(2, 0);
    const auto response = client.Get(path);
    if (response && response->status == 200) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

void InstallStopHandler(std::atomic<bool>& flag) {
  g_stopFlag = &flag;
#ifdef _WIN32
  SetConsoleCtrlHandler(ConsoleHandler, TRUE);
#else
  std::signal(SIGINT, PosixHandler);
  std::signal(SIGTERM, PosixHandler);
#endif
}

} // namespace maya::launcher
