import 'package:flutter_test/flutter_test.dart';
import 'package:llm_chat/services/settings_service.dart';

void main() {
  test('default server url is set', () {
    expect(AppSettings.defaultServerUrl(), isNotEmpty);
  });
}
