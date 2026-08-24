import 'dart:convert';



import 'package:http/http.dart' as http;



class ChatApiException implements Exception {

  ChatApiException(this.message);



  final String message;



  @override

  String toString() => message;

}



class HealthInfo {

  const HealthInfo({

    required this.loaded,

    required this.stage,

    this.backend,

    this.model,

  });



  final bool loaded;

  final String stage;

  final String? backend;

  final String? model;

}



class ServerConfig {

  const ServerConfig({

    required this.backend,

    required this.model,

    required this.localAvailable,

    required this.openAiAvailable,

    required this.openAiModel,

  });



  final String backend;

  final String model;

  final bool localAvailable;

  final bool openAiAvailable;

  final String openAiModel;

}



class ChatCompletion {
  const ChatCompletion({required this.text, this.intent});

  final String text;
  final String? intent;
}

class ChatApi {

  ChatApi(this.baseUrl);



  final String baseUrl;



  Uri _uri(String path) {

    final normalized = baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl;

    return Uri.parse('$normalized$path');

  }



  Future<HealthInfo> health() async {

    final response = await http.get(_uri('/api/v1/health')).timeout(const Duration(seconds: 8));

    if (response.statusCode != 200) {

      throw ChatApiException('health status ${response.statusCode}');

    }



    final json = jsonDecode(response.body) as Map<String, dynamic>;

    return HealthInfo(

      loaded: json['loaded'] == true,

      stage: json['stage']?.toString() ?? '',

      backend: json['backend']?.toString(),

      model: json['model']?.toString(),

    );

  }



  Future<ServerConfig> getConfig() async {

    final response = await http.get(_uri('/api/v1/config')).timeout(const Duration(seconds: 8));

    if (response.statusCode != 200) {

      throw ChatApiException('config status ${response.statusCode}');

    }



    final json = jsonDecode(response.body) as Map<String, dynamic>;

    return ServerConfig(

      backend: json['backend']?.toString() ?? 'openai',

      model: json['model']?.toString() ?? '',

      localAvailable: json['local_available'] == true,

      openAiAvailable: json['openai_available'] == true,

      openAiModel: json['openai_model']?.toString() ?? 'llama3.2',

    );

  }



  Future<ServerConfig> setConfig({

    required String backend,

    required String openAiModel,

  }) async {

    final response = await http

        .post(

          _uri('/api/v1/config'),

          headers: {'Content-Type': 'application/json'},

          body: jsonEncode({

            'backend': backend,

            'openai_model': openAiModel,

          }),

        )

        .timeout(const Duration(seconds: 15));



    if (response.statusCode != 200) {

      final json = jsonDecode(response.body) as Map<String, dynamic>;

      throw ChatApiException(json['error']?.toString() ?? 'config status ${response.statusCode}');

    }



    final json = jsonDecode(response.body) as Map<String, dynamic>;

    return ServerConfig(

      backend: json['backend']?.toString() ?? backend,

      model: json['model']?.toString() ?? '',

      localAvailable: json['local_available'] == true,

      openAiAvailable: json['openai_available'] == true,

      openAiModel: json['openai_model']?.toString() ?? openAiModel,

    );

  }



  Future<ChatCompletion> complete({

    required String prompt,

    required int maxTokens,

    required double temperature,

    required bool greedy,

  }) async {

    final response = await http

        .post(

          _uri('/api/v1/chat'),

          headers: {'Content-Type': 'application/json'},

          body: jsonEncode({

            'prompt': prompt,

            'max_tokens': maxTokens,

            'temperature': temperature,

            'greedy': greedy,

          }),

        )

        .timeout(const Duration(minutes: 2));



    if (response.statusCode != 200 && response.statusCode != 400) {

      throw ChatApiException('chat status ${response.statusCode}');

    }



    final json = jsonDecode(response.body) as Map<String, dynamic>;

    final error = json['error'];

    if (error != null && error.toString().isNotEmpty) {

      throw ChatApiException(error.toString());

    }



    return ChatCompletion(
      text: json['text']?.toString() ?? '',
      intent: json['intent']?.toString(),
    );

  }



  Future<void> reset() async {

    final response = await http.post(_uri('/api/v1/reset')).timeout(const Duration(seconds: 8));

    if (response.statusCode != 200) {

      throw ChatApiException('reset status ${response.statusCode}');

    }

  }

}


