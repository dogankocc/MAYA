import 'dart:io';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum ModelBackend { local, openai }

class AppSettings {
  const AppSettings({
    required this.serverUrl,
    required this.maxTokens,
    required this.temperature,
    required this.greedy,
    required this.themeMode,
    required this.locale,
    required this.modelBackend,
    required this.openAiModel,
  });

  final String serverUrl;
  final int maxTokens;
  final double temperature;
  final bool greedy;
  final ThemeMode themeMode;
  final Locale? locale;
  final ModelBackend modelBackend;
  final String openAiModel;

  static const openAiModelOptions = ['llama3.2', 'mistral', 'gemma2'];

  AppSettings copyWith({
    String? serverUrl,
    int? maxTokens,
    double? temperature,
    bool? greedy,
    ThemeMode? themeMode,
    Locale? locale,
    ModelBackend? modelBackend,
    String? openAiModel,
  }) {
    return AppSettings(
      serverUrl: serverUrl ?? this.serverUrl,
      maxTokens: maxTokens ?? this.maxTokens,
      temperature: temperature ?? this.temperature,
      greedy: greedy ?? this.greedy,
      themeMode: themeMode ?? this.themeMode,
      locale: locale ?? this.locale,
      modelBackend: modelBackend ?? this.modelBackend,
      openAiModel: openAiModel ?? this.openAiModel,
    );
  }

  static String defaultServerUrl() {
    if (Platform.isAndroid) {
      return 'http://10.0.2.2:8765';
    }
    return 'http://127.0.0.1:8765';
  }

  static const AppSettings defaults = AppSettings(
    serverUrl: 'http://127.0.0.1:8765',
    maxTokens: 128,
    temperature: 0.3,
    greedy: true,
    themeMode: ThemeMode.system,
    locale: null,
    modelBackend: ModelBackend.local,
    openAiModel: 'llama3.2',
  );
}

class SettingsService {
  static const _serverUrlKey = 'server_url';
  static const _maxTokensKey = 'max_tokens';
  static const _temperatureKey = 'temperature';
  static const _greedyKey = 'greedy';
  static const _themeModeKey = 'theme_mode';
  static const _localeKey = 'locale';
  static const _modelBackendKey = 'model_backend';
  static const _openAiModelKey = 'openai_model';

  Future<AppSettings> load() async {
    final prefs = await SharedPreferences.getInstance();
    final themeIndex = prefs.getInt(_themeModeKey) ?? ThemeMode.system.index;
    final localeCode = prefs.getString(_localeKey);
    final backendIndex = prefs.getInt(_modelBackendKey) ?? ModelBackend.openai.index;
    final openAiModel = prefs.getString(_openAiModelKey) ?? AppSettings.defaults.openAiModel;

    return AppSettings(
      serverUrl: prefs.getString(_serverUrlKey) ?? AppSettings.defaultServerUrl(),
      maxTokens: prefs.getInt(_maxTokensKey) ?? AppSettings.defaults.maxTokens,
      temperature: prefs.getDouble(_temperatureKey) ?? AppSettings.defaults.temperature,
      greedy: prefs.getBool(_greedyKey) ?? AppSettings.defaults.greedy,
      themeMode: ThemeMode.values[themeIndex.clamp(0, ThemeMode.values.length - 1)],
      locale: localeCode == null ? null : Locale(localeCode),
      modelBackend: ModelBackend.values[backendIndex.clamp(0, ModelBackend.values.length - 1)],
      openAiModel: AppSettings.openAiModelOptions.contains(openAiModel)
          ? openAiModel
          : AppSettings.defaults.openAiModel,
    );
  }

  Future<void> save(AppSettings settings) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_serverUrlKey, settings.serverUrl);
    await prefs.setInt(_maxTokensKey, settings.maxTokens);
    await prefs.setDouble(_temperatureKey, settings.temperature);
    await prefs.setBool(_greedyKey, settings.greedy);
    await prefs.setInt(_themeModeKey, settings.themeMode.index);
    await prefs.setInt(_modelBackendKey, settings.modelBackend.index);
    await prefs.setString(_openAiModelKey, settings.openAiModel);
    if (settings.locale == null) {
      await prefs.remove(_localeKey);
    } else {
      await prefs.setString(_localeKey, settings.locale!.languageCode);
    }
  }
}
