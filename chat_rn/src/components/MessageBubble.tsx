import { StyleSheet, Text, View } from 'react-native';

import type { ChatMessage } from '@/models/chatMessage';
import { isUserMessage } from '@/models/chatMessage';
import type { AppColors } from '@/theme/colors';

interface MessageBubbleProps {
  message: ChatMessage;
  userLabel: string;
  assistantLabel: string;
  colors: AppColors;
}

export function MessageBubble({
  message,
  userLabel,
  assistantLabel,
  colors,
}: MessageBubbleProps) {
  const isUser = isUserMessage(message);

  const bubbleColor = message.isError
    ? colors.errorContainer
    : isUser
      ? colors.primaryContainer
      : colors.surfaceHigh;

  const textColor = message.isError
    ? colors.onErrorContainer
    : isUser
      ? colors.onPrimaryContainer
      : colors.onSurface;

  return (
    <View style={[styles.row, isUser ? styles.rowUser : styles.rowAssistant]}>
      <View style={styles.column}>
        <Text style={[styles.label, { color: colors.outline }]}>
          {isUser ? userLabel : assistantLabel}
        </Text>
        <View
          style={[
            styles.bubble,
            { backgroundColor: bubbleColor },
            isUser ? styles.bubbleUser : styles.bubbleAssistant,
          ]}
        >
          <Text selectable style={[styles.text, { color: textColor }]}>
            {message.text}
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    maxWidth: '82%',
  },
  rowUser: {
    alignSelf: 'flex-end',
  },
  rowAssistant: {
    alignSelf: 'flex-start',
  },
  column: {
    gap: 4,
  },
  label: {
    fontSize: 12,
  },
  bubble: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 16,
  },
  bubbleUser: {
    borderBottomRightRadius: 4,
  },
  bubbleAssistant: {
    borderBottomLeftRadius: 4,
  },
  text: {
    fontSize: 16,
    lineHeight: 22,
  },
});
