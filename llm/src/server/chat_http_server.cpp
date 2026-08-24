#include "llm/server/chat_http_server.hpp"

#include <thread>

#include "httplib.h"
#include "server/json_util.hpp"

namespace llm::server {

ChatHttpServer::ChatHttpServer(std::unique_ptr<HybridChatBackend> backend,
                               std::unique_ptr<AutoLearnService> autoLearn)
    : backend_(std::move(backend)), autoLearn_(std::move(autoLearn)) {}

ChatResponse ChatHttpServer::HandleChat(const ChatRequest& request) {
  if (!backend_) {
    return ChatResponse{.text = "", .error = "backend is not configured"};
  }

  std::lock_guard<std::mutex> lock(mutex_);
  return backend_->Complete(request);
}

Status ChatHttpServer::HandleReset() {
  if (!backend_) {
    return Status::Fail(ErrorCode::Internal, "backend is not configured");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  return backend_->Reset();
}

Status ChatHttpServer::Run(const std::string& host, const int port) {
  httplib::Server http;
  stopRequested_ = false;

  http.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                            {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                            {"Access-Control-Allow-Headers", "Content-Type"}});

  http.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& res) {
    res.status = 204;
  });

  http.Get("/api/v1/health", [this](const httplib::Request&, httplib::Response& res) {
    const BackendInfo info = backend_ ? backend_->GetInfo() : BackendInfo{};
    const AutoLearnStatus autoLearnStatus = autoLearn_ ? autoLearn_->GetStatus() : AutoLearnStatus{};
    res.set_content(detail::BuildHealthJson(info, autoLearnStatus), "application/json");
  });

  http.Get("/api/v1/config", [this](const httplib::Request&, httplib::Response& res) {
    if (!backend_) {
      res.status = 500;
      res.set_content("{\"error\":\"backend is not configured\"}", "application/json");
      return;
    }
    res.set_content(detail::BuildConfigJson(backend_->GetConfig()), "application/json");
  });

  http.Post("/api/v1/config", [this](const httplib::Request& req, httplib::Response& res) {
    if (!backend_) {
      res.status = 500;
      res.set_content("{\"ok\":false,\"error\":\"backend is not configured\"}", "application/json");
      return;
    }

    const auto parsed = detail::ParseConfigRequest(req.body);
    if (!parsed.has_value()) {
      res.status = 400;
      res.set_content("{\"ok\":false,\"error\":\"invalid json body\"}", "application/json");
      return;
    }

    const Status backendStatus = backend_->SetActiveBackend(parsed->backend);
    if (!backendStatus.IsOk()) {
      res.status = 400;
      res.set_content("{\"ok\":false,\"error\":\"" + detail::JsonEscape(backendStatus.Message()) + "\"}",
                      "application/json");
      return;
    }

    if (!parsed->openAiModel.empty()) {
      const Status modelStatus = backend_->SetOpenAiModel(parsed->openAiModel);
      if (!modelStatus.IsOk()) {
        res.status = 400;
        res.set_content("{\"ok\":false,\"error\":\"" + detail::JsonEscape(modelStatus.Message()) + "\"}",
                        "application/json");
        return;
      }
    }

    backend_->Reset();
    res.set_content(detail::BuildConfigJson(backend_->GetConfig()), "application/json");
  });

  http.Post("/api/v1/reset", [this](const httplib::Request&, httplib::Response& res) {
    const Status status = HandleReset();
    if (!status.IsOk()) {
      res.status = 500;
      res.set_content("{\"ok\":false,\"error\":\"" + detail::JsonEscape(status.Message()) + "\"}",
                      "application/json");
      return;
    }
    res.set_content(detail::BuildOkJson(), "application/json");
  });

  http.Post("/api/v1/chat", [this](const httplib::Request& req, httplib::Response& res) {
    const auto parsed = detail::ParseChatRequest(req.body);
    if (!parsed.has_value()) {
      res.status = 400;
      res.set_content("{\"text\":\"\",\"error\":\"invalid json body\"}", "application/json");
      return;
    }

    const ChatResponse response = HandleChat(parsed.value());
    if (autoLearn_ && response.allowAutoLearn && response.error.empty() && !response.text.empty()) {
      autoLearn_->OnChatCompleted(parsed->prompt, response.text, response.intent);
    }
    if (!response.error.empty() && response.text.empty()) {
      res.status = 400;
    }
    res.set_content(detail::BuildChatResponseJson(response), "application/json");
  });

  if (!http.bind_to_port(host.c_str(), port)) {
    return Status::Fail(ErrorCode::Internal, "failed to bind HTTP port");
  }

  std::thread serverThread([&http]() { http.listen_after_bind(); });

  while (!stopRequested_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  http.stop();
  if (serverThread.joinable()) {
    serverThread.join();
  }

  if (autoLearn_) {
    autoLearn_->Shutdown();
  }

  return Status::Ok();
}

void ChatHttpServer::RequestStop() {
  stopRequested_ = true;
}

} // namespace llm::server
