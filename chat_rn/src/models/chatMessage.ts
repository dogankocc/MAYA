export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: Date;
  isError?: boolean;
}

export function isUserMessage(message: ChatMessage): boolean {
  return message.role === 'user';
}
