import 'package:flutter/material.dart';

import '../models/chat_message.dart';
import '../services/chat_api.dart';
import '../services/settings_service.dart';

enum ServerConnectionState { unknown, connected, disconnected }

class ChatController extends ChangeNotifier {
  ChatController(this._settingsService);

  final SettingsService _settingsService;

  AppSettings settings = AppSettings(
    serverUrl: AppSettings.defaultServerUrl(),
    maxTokens: AppSettings.defaults.maxTokens,
    temperature: AppSettings.defaults.temperature,
    greedy: AppSettings.defaults.greedy,
    themeMode: AppSettings.defaults.themeMode,
    locale: AppSettings.defaults.locale,
    modelBackend: AppSettings.defaults.modelBackend,
    openAiModel: AppSettings.defaults.openAiModel,
  );

  final List<ChatMessage> messages = [];
  ServerConnectionState connectionState = ServerConnectionState.unknown;
  bool isSending = false;
  String? serverStage;
  String? serverModel;

  int _messageCounter = 0;

  Future<void> initialize() async {
    settings = await _settingsService.load();
    try {
      await _applyBackendConfig(settings);
    } catch (_) {
      // server may be offline during startup
    }
    await refreshHealth();
  }

  Future<void> _applyBackendConfig(AppSettings config) async {
    final api = ChatApi(config.serverUrl);
    final serverConfig = await api.setConfig(
      backend: config.modelBackend.name,
      openAiModel: config.openAiModel,
    );
    serverModel = serverConfig.model;
    serverStage = serverConfig.backend == 'local' ? 'local-mini' : 'ollama-compatible';
  }

  Future<void> updateSettings(AppSettings newSettings) async {
    final backendChanged = newSettings.modelBackend != settings.modelBackend ||
        newSettings.openAiModel != settings.openAiModel ||
        newSettings.serverUrl != settings.serverUrl;

    settings = newSettings;
    await _settingsService.save(settings);
    notifyListeners();

    try {
      await _applyBackendConfig(settings);
      if (backendChanged) {
        messages.clear();
      }
    } catch (_) {
      connectionState = ServerConnectionState.disconnected;
      notifyListeners();
      return;
    }

    await refreshHealth();
  }

  Future<void> refreshHealth() async {
    connectionState = ServerConnectionState.unknown;
    notifyListeners();

    try {
      final api = ChatApi(settings.serverUrl);
      final health = await api.health();
      serverStage = health.stage;
      serverModel = health.model ?? serverModel;
      connectionState = health.loaded ? ServerConnectionState.connected : ServerConnectionState.disconnected;
    } catch (_) {
      connectionState = ServerConnectionState.disconnected;
      serverStage = null;
      serverModel = null;
    }

    notifyListeners();
  }

  Future<void> resetConversation() async {
    final api = ChatApi(settings.serverUrl);
    await api.reset();
    messages.clear();
    notifyListeners();
  }

  Future<void> sendMessage(String text) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty || isSending) {
      return;
    }

    isSending = true;
    _messageCounter += 1;
    messages.add(
      ChatMessage(
        id: 'u$_messageCounter',
        role: MessageRole.user,
        text: trimmed,
        timestamp: DateTime.now(),
      ),
    );
    notifyListeners();

    try {
      final api = ChatApi(settings.serverUrl);
      final reply = await api.complete(
        prompt: trimmed,
        maxTokens: settings.maxTokens,
        temperature: settings.temperature,
        greedy: settings.greedy,
      );

      _messageCounter += 1;
      messages.add(
        ChatMessage(
          id: 'a$_messageCounter',
          role: MessageRole.assistant,
          text: reply.text.isEmpty ? '...' : reply.text,
          timestamp: DateTime.now(),
        ),
      );
      connectionState = ServerConnectionState.connected;
    } on ChatApiException catch (error) {
      _messageCounter += 1;
      messages.add(
        ChatMessage(
          id: 'e$_messageCounter',
          role: MessageRole.assistant,
          text: error.message,
          timestamp: DateTime.now(),
          isError: true,
        ),
      );
      connectionState = ServerConnectionState.disconnected;
    } catch (error) {
      _messageCounter += 1;
      messages.add(
        ChatMessage(
          id: 'e$_messageCounter',
          role: MessageRole.assistant,
          text: error.toString(),
          timestamp: DateTime.now(),
          isError: true,
        ),
      );
      connectionState = ServerConnectionState.disconnected;
    } finally {
      isSending = false;
      notifyListeners();
    }
  }
}
