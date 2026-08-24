import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../l10n/app_localizations.dart';
import '../providers/chat_controller.dart';
import '../services/settings_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _serverController;
  late int _maxTokens;
  late double _temperature;
  late bool _greedy;
  late ThemeMode _themeMode;
  late ModelBackend _modelBackend;
  late String _openAiModel;
  String? _localeCode;
  String? _testResult;

  @override
  void initState() {
    super.initState();
    final settings = context.read<ChatController>().settings;
    _serverController = TextEditingController(text: settings.serverUrl);
    _maxTokens = settings.maxTokens;
    _temperature = settings.temperature;
    _greedy = settings.greedy;
    _themeMode = settings.themeMode;
    _modelBackend = settings.modelBackend;
    _openAiModel = settings.openAiModel;
    _localeCode = settings.locale?.languageCode;
  }

  @override
  void dispose() {
    _serverController.dispose();
    super.dispose();
  }

  AppSettings _buildSettings() {
    return AppSettings(
      serverUrl: _serverController.text.trim(),
      maxTokens: _maxTokens,
      temperature: _temperature,
      greedy: _greedy,
      themeMode: _themeMode,
      locale: _localeCode == null ? null : Locale(_localeCode!),
      modelBackend: _modelBackend,
      openAiModel: _openAiModel,
    );
  }

  Future<void> _save() async {
    final controller = context.read<ChatController>();
    await controller.updateSettings(_buildSettings());
    if (mounted) {
      Navigator.of(context).pop();
    }
  }

  Future<void> _testConnection() async {
    setState(() => _testResult = null);
    final controller = context.read<ChatController>();
    await controller.updateSettings(_buildSettings());
    setState(() {
      _testResult = controller.connectionState == ServerConnectionState.connected
          ? AppLocalizations.of(context)!.connected
          : AppLocalizations.of(context)!.errorHealth;
    });
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;

    return Scaffold(
      appBar: AppBar(title: Text(l10n.settings)),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _serverController,
            decoration: InputDecoration(
              labelText: l10n.serverUrl,
              hintText: AppSettings.defaultServerUrl(),
            ),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: _testConnection,
            icon: const Icon(Icons.wifi_tethering_rounded),
            label: Text(l10n.testConnection),
          ),
          if (_testResult != null) ...[
            const SizedBox(height: 8),
            Text(_testResult!, style: Theme.of(context).textTheme.bodySmall),
          ],
          const SizedBox(height: 20),
          Text(l10n.modelSource, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          SegmentedButton<ModelBackend>(
            segments: [
              ButtonSegment(value: ModelBackend.local, label: Text(l10n.modelLocal)),
              ButtonSegment(value: ModelBackend.openai, label: Text(l10n.modelOpenAi)),
            ],
            selected: {_modelBackend},
            onSelectionChanged: (value) => setState(() => _modelBackend = value.first),
          ),
          if (_modelBackend == ModelBackend.openai) ...[
            const SizedBox(height: 16),
            Text(l10n.openAiModel, style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: AppSettings.openAiModelOptions.map((model) {
                final selected = _openAiModel == model;
                return FilterChip(
                  label: Text(model),
                  selected: selected,
                  onSelected: (_) => setState(() => _openAiModel = model),
                );
              }).toList(),
            ),
          ],
          const SizedBox(height: 20),
          Text(l10n.maxTokens, style: Theme.of(context).textTheme.titleSmall),
          Slider(
            value: _maxTokens.toDouble(),
            min: 8,
            max: 256,
            divisions: 31,
            label: '$_maxTokens',
            onChanged: (value) => setState(() => _maxTokens = value.round()),
          ),
          Text(l10n.temperature, style: Theme.of(context).textTheme.titleSmall),
          Slider(
            value: _temperature,
            min: 0.1,
            max: 1.5,
            divisions: 14,
            label: _temperature.toStringAsFixed(1),
            onChanged: (value) => setState(() => _temperature = value),
          ),
          SwitchListTile(
            title: Text(l10n.greedyMode),
            value: _greedy,
            onChanged: (value) => setState(() => _greedy = value),
          ),
          const Divider(height: 32),
          Text(l10n.theme, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          SegmentedButton<ThemeMode>(
            segments: [
              ButtonSegment(value: ThemeMode.system, label: Text(l10n.themeSystem)),
              ButtonSegment(value: ThemeMode.light, label: Text(l10n.themeLight)),
              ButtonSegment(value: ThemeMode.dark, label: Text(l10n.themeDark)),
            ],
            selected: {_themeMode},
            onSelectionChanged: (value) => setState(() => _themeMode = value.first),
          ),
          const SizedBox(height: 20),
          Text(l10n.language, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          SegmentedButton<String?>(
            segments: const [
              ButtonSegment(value: null, label: Text('Auto')),
              ButtonSegment(value: 'tr', label: Text('TR')),
              ButtonSegment(value: 'en', label: Text('EN')),
            ],
            selected: {_localeCode},
            onSelectionChanged: (value) => setState(() => _localeCode = value.first),
          ),
          const SizedBox(height: 32),
          FilledButton(onPressed: _save, child: Text(l10n.save)),
        ],
      ),
    );
  }
}
