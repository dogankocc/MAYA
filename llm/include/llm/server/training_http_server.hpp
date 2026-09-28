#pragma once

#include <atomic>
#include <string>

#include "llm/core/status.hpp"
#include "llm/server/training_service.hpp"

namespace llm::server {

// REST API + static web UI for the model trainer. Runs in its own process so the
// chat server is never affected by training load.
class TrainingHttpServer {
public:
  TrainingHttpServer(TrainingService& service, std::string webRoot = "trainer_web");

  [[nodiscard]] Status Run(const std::string& host, int port);

  void RequestStop();

private:
  TrainingService& service_;
  std::string webRoot_;
  std::atomic<bool> stopRequested_{false};
};

} // namespace llm::server
