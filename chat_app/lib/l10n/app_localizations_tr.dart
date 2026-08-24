// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Turkish (`tr`).
class AppLocalizationsTr extends AppLocalizations {
  AppLocalizationsTr([String locale = 'tr']) : super(locale);

  @override
  String get appTitle => 'LLM Sohbet';

  @override
  String get chatHint => 'Mesaj yazın...';

  @override
  String get send => 'Gönder';

  @override
  String get settings => 'Ayarlar';

  @override
  String get serverUrl => 'Sunucu adresi';

  @override
  String get maxTokens => 'Maks. token';

  @override
  String get temperature => 'Sıcaklık';

  @override
  String get greedyMode => 'Greedy (deterministik)';

  @override
  String get resetChat => 'Sohbeti sıfırla';

  @override
  String get connecting => 'Bağlanıyor...';

  @override
  String get connected => 'Bağlı';

  @override
  String get disconnected => 'Bağlantı yok';

  @override
  String get errorSend => 'Mesaj gönderilemedi';

  @override
  String get errorHealth => 'Sunucuya ulaşılamıyor';

  @override
  String get welcomeTitle => 'Merhaba!';

  @override
  String get welcomeBody => 'Yerel LLM modelinizle sohbete başlayın.';

  @override
  String get you => 'Sen';

  @override
  String get assistant => 'Asistan';

  @override
  String get theme => 'Tema';

  @override
  String get themeSystem => 'Sistem';

  @override
  String get themeLight => 'Açık';

  @override
  String get themeDark => 'Koyu';

  @override
  String get language => 'Dil';

  @override
  String get save => 'Kaydet';

  @override
  String get testConnection => 'Bağlantıyı test et';

  @override
  String get modelSource => 'Model kaynağı';

  @override
  String get modelLocal => 'Yerel model';

  @override
  String get modelOpenAi => 'OpenAI / Ollama';

  @override
  String get openAiModel => 'Ollama modeli';
}
