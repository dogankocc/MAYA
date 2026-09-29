#include "llm/server/training_http_server.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>

#include "httplib.h"
#include "llm/server/model_manager.hpp"
#include "server/json_util.hpp"

namespace llm::server {

namespace {

using detail::ExtractSizeField;
using detail::ExtractStringField;
using detail::JsonEscape;

[[nodiscard]] std::string Number(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(9) << value;
  return stream.str();
}

[[nodiscard]] std::string Quoted(const std::string& value) {
  return "\"" + JsonEscape(value) + "\"";
}

[[nodiscard]] std::string ErrorJson(const std::string& message) {
  return "{\"ok\":false,\"error\":" + Quoted(message) + "}";
}

void Reply(httplib::Response& res, const std::string& json, const int status = 200) {
  res.status = status;
  res.set_content(json, "application/json");
}

void ReplyStatus(httplib::Response& res, const Status& status) {
  if (status.IsOk()) {
    Reply(res, detail::BuildOkJson());
  } else {
    Reply(res, ErrorJson(status.Message()), 400);
  }
}

[[nodiscard]] std::string DatasetsJson(const std::vector<DatasetInfo>& datasets) {
  std::string json = "{\"datasets\":[";
  for (std::size_t i = 0; i < datasets.size(); ++i) {
    const DatasetInfo& dataset = datasets[i];
    if (i > 0) {
      json += ',';
    }
    json += "{\"name\":" + Quoted(dataset.name) + ",\"path\":" + Quoted(dataset.path) +
            ",\"description\":" + Quoted(dataset.description) + ",\"size_bytes\":" + std::to_string(dataset.sizeBytes) +
            ",\"sample_count\":" + std::to_string(dataset.sampleCount) + "}";
  }
  return json + "]}";
}

[[nodiscard]] std::string StorageJson(const StorageInfo& storage) {
  return "{\"free_bytes\":" + std::to_string(storage.freeBytes) + ",\"total_bytes\":" +
         std::to_string(storage.totalBytes) + ",\"model_bytes\":" + std::to_string(storage.modelBytes) +
         ",\"quant_bytes\":" + std::to_string(storage.quantBytes) + ",\"tokenizer_bytes\":" +
         std::to_string(storage.tokenizerBytes) + ",\"temp_checkpoint_bytes\":" +
         std::to_string(storage.tempCheckpointBytes) + "}";
}

[[nodiscard]] std::string SessionJson(const std::optional<training::TrainingSessionState>& session) {
  if (!session.has_value()) {
    return "null";
  }
  return training::SerializeSessionState(*session);
}

[[nodiscard]] std::string StatusJson(const TrainingStatusSnapshot& status, const StorageInfo& storage,
                                     const std::optional<training::TrainingSessionState>& pending) {
  std::string json = "{\"state\":" + Quoted(TrainingJobStateName(status.state));
  json += ",\"step\":" + std::to_string(status.step);
  json += ",\"total_steps\":" + std::to_string(status.totalSteps);
  json += ",\"loss\":" + Number(status.loss);
  json += ",\"resumed_from\":" + std::to_string(status.resumedFromStep);
  json += ",\"elapsed_seconds\":" + Number(status.elapsedSeconds);
  json += ",\"steps_per_second\":" + Number(status.stepsPerSecond);
  json += ",\"message\":" + Quoted(status.message);
  json += std::string(",\"downloading\":") + (status.downloading ? "true" : "false");
  json += ",\"active_config\":" +
          (status.activeConfig ? "{" + training::SerializeTrainingJobConfigFields(*status.activeConfig) + "}"
                               : std::string("null"));
  json += ",\"storage\":" + StorageJson(storage);
  json += ",\"pending_session\":" + SessionJson(pending);
  return json + "}";
}

[[nodiscard]] std::string LogsJson(const std::vector<LogEntry>& entries) {
  std::string json = "{\"entries\":[";
  std::uint64_t lastSeq = 0;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const LogEntry& entry = entries[i];
    if (i > 0) {
      json += ',';
    }
    json += "{\"seq\":" + std::to_string(entry.seq) + ",\"time\":" + std::to_string(entry.unixTime) +
            ",\"text\":" + Quoted(entry.text) + "}";
    lastSeq = entry.seq;
  }
  return json + "],\"last_seq\":" + std::to_string(lastSeq) + "}";
}

[[nodiscard]] std::string PresetJson(const ModelPreset preset) {
  CreateModelRequest request;
  request.learningRate = 0.0f;
  request.batchSize = 0;
  ApplyModelPreset(preset, request);
  return "{\"id\":" + Quoted(std::to_string(static_cast<int>(preset))) + ",\"name\":" + Quoted(GetPresetName(preset)) +
         ",\"description\":" + Quoted(GetPresetDescription(preset)) +
         ",\"num_layers\":" + std::to_string(request.numLayers) + ",\"hidden_dim\":" + std::to_string(request.hiddenDim) +
         ",\"num_heads\":" + std::to_string(request.numHeads) + ",\"num_kv_heads\":" + std::to_string(request.numKvHeads) +
         ",\"intermediate_dim\":" + std::to_string(request.intermediateDim) +
         ",\"max_seq_len\":" + std::to_string(request.maxSeqLen) + ",\"vocab_size\":" + std::to_string(request.vocabSize) +
         ",\"learning_rate\":" + Number(request.learningRate) +
         ",\"batch_size\":" + std::to_string(request.batchSize) + "}";
}

[[nodiscard]] std::string PresetsJson() {
  std::string json = "{\"presets\":[";
  bool first = true;
  for (const ModelPreset preset : {ModelPreset::Tiny, ModelPreset::Small, ModelPreset::Medium, ModelPreset::Large,
                                   ModelPreset::XLarge}) {
    if (!first) {
      json += ',';
    }
    first = false;
    json += PresetJson(preset);
  }
  return json + "],\"defaults\":{" + training::SerializeTrainingJobConfigFields(training::DefaultTrainingJobConfig()) +
         "}}";
}

} // namespace

TrainingHttpServer::TrainingHttpServer(TrainingService& service, std::string webRoot)
    : service_(service), webRoot_(std::move(webRoot)) {}

Status TrainingHttpServer::Run(const std::string& host, const int port) {
  httplib::Server http;
  stopRequested_ = false;

  http.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                            {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                            {"Access-Control-Allow-Headers", "Content-Type"},
                            {"Cache-Control", "no-store"}});
  http.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });

  http.Get("/api/v1/train/datasets", [this](const httplib::Request&, httplib::Response& res) {
    Reply(res, DatasetsJson(service_.ListDatasets()));
  });

  http.Get("/api/v1/train/presets", [](const httplib::Request&, httplib::Response& res) {
    Reply(res, PresetsJson());
  });

  http.Get("/api/v1/train/status", [this](const httplib::Request&, httplib::Response& res) {
    Reply(res, StatusJson(service_.GetStatus(), service_.GetStorage(), service_.PendingSession()));
  });

  http.Get("/api/v1/train/logs", [this](const httplib::Request& req, httplib::Response& res) {
    std::uint64_t since = 0;
    if (req.has_param("since")) {
      try {
        since = std::stoull(req.get_param_value("since"));
      } catch (...) {
        since = 0;
      }
    }
    Reply(res, LogsJson(service_.GetLogs(since)));
  });

  http.Post("/api/v1/train/start", [this](const httplib::Request& req, httplib::Response& res) {
    const training::TrainingJobConfig config =
        training::ParseTrainingJobConfig(req.body, training::DefaultTrainingJobConfig());
    if (config.datasetPaths.empty() && !config.includeBuiltin) {
      Reply(res, ErrorJson("select at least one dataset"), 400);
      return;
    }
    ReplyStatus(res, service_.Start(config));
  });

  http.Post("/api/v1/train/stop", [this](const httplib::Request&, httplib::Response& res) {
    service_.RequestStop();
    Reply(res, detail::BuildOkJson());
  });

  http.Post("/api/v1/train/session/clear", [this](const httplib::Request&, httplib::Response& res) {
    ReplyStatus(res, service_.ClearPendingSession());
  });

  http.Post("/api/v1/train/download", [this](const httplib::Request& req, httplib::Response& res) {
    const auto script = ExtractStringField(req.body, "script");
    ReplyStatus(res, service_.StartDownload(script.value_or("code")));
  });

  if (!http.set_mount_point("/", webRoot_)) {
    return Status::Fail(ErrorCode::NotFound, "trainer web root not found: " + webRoot_);
  }

  if (!http.bind_to_port(host.c_str(), port)) {
    return Status::Fail(ErrorCode::Internal, "failed to bind HTTP port " + std::to_string(port));
  }

  std::thread serverThread([&http]() { http.listen_after_bind(); });
  while (!stopRequested_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  http.stop();
  if (serverThread.joinable()) {
    serverThread.join();
  }
  service_.Shutdown();
  return Status::Ok();
}

void TrainingHttpServer::RequestStop() {
  stopRequested_.store(true);
}

} // namespace llm::server
