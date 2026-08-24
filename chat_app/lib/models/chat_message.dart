enum MessageRole { user, assistant }

class ChatMessage {
  const ChatMessage({
    required this.id,
    required this.role,
    required this.text,
    required this.timestamp,
    this.isError = false,
  });

  final String id;
  final MessageRole role;
  final String text;
  final DateTime timestamp;
  final bool isError;

  bool get isUser => role == MessageRole.user;
}
