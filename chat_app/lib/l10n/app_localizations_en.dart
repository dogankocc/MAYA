// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'LLM Chat';

  @override
  String get chatHint => 'Type a message...';

  @override
  String get send => 'Send';

  @override
  String get settings => 'Settings';

  @override
  String get serverUrl => 'Server URL';

  @override
  String get maxTokens => 'Max tokens';

  @override
  String get temperature => 'Temperature';

  @override
  String get greedyMode => 'Greedy (deterministic)';

  @override
  String get resetChat => 'Reset conversation';

  @override
  String get connecting => 'Connecting...';

  @override
  String get connected => 'Connected';

  @override
  String get disconnected => 'Disconnected';

  @override
  String get errorSend => 'Failed to send message';

  @override
  String get errorHealth => 'Cannot reach server';

  @override
  String get welcomeTitle => 'Hello!';

  @override
  String get welcomeBody => 'Start chatting with your local LLM model.';

  @override
  String get you => 'You';

  @override
  String get assistant => 'Assistant';

  @override
  String get theme => 'Theme';

  @override
  String get themeSystem => 'System';

  @override
  String get themeLight => 'Light';

  @override
  String get themeDark => 'Dark';

  @override
  String get language => 'Language';

  @override
  String get save => 'Save';

  @override
  String get testConnection => 'Test connection';

  @override
  String get modelSource => 'Model source';

  @override
  String get modelLocal => 'Local model';

  @override
  String get modelOpenAi => 'OpenAI / Ollama';

  @override
  String get openAiModel => 'Ollama model';
}
