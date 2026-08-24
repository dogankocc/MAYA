import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import { MessageBubble } from '@/components/MessageBubble';
import { WelcomeCard } from '@/components/WelcomeCard';
import { useChat } from '@/context/ChatContext';
import type { ChatMessage } from '@/models/chatMessage';
import type { RootStackParamList } from '@/navigation/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Chat'>;

export function ChatScreen({ navigation }: Props) {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const {
    messages,
    connectionState,
    serverStage,
    serverModel,
    isSending,
    colors,
    resetConversation,
    sendMessage,
  } = useChat();

  const [input, setInput] = useState('');
  const listRef = useRef<FlatList<ChatMessage>>(null);

  const scrollToBottom = useCallback(() => {
    if (messages.length === 0) {
      return;
    }
    requestAnimationFrame(() => {
      listRef.current?.scrollToEnd({ animated: true });
    });
  }, [messages.length]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const statusText = (() => {
    if (connectionState === 'connected') {
      const detail = serverModel ?? serverStage;
      return detail ? `${t('connected')} · ${detail}` : t('connected');
    }
    if (connectionState === 'disconnected') {
      return t('disconnected');
    }
    return t('connecting');
  })();

  const handleSend = async () => {
    const text = input;
    setInput('');
    await sendMessage(text);
  };

  return (
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: colors.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={insets.top}
    >
      <View style={[styles.header, { borderBottomColor: colors.border }]}>
        <View style={styles.headerText}>
          <Text style={[styles.title, { color: colors.onSurface }]}>{t('appTitle')}</Text>
          <Text style={[styles.status, { color: colors.onSurfaceVariant }]}>
            {statusText}
          </Text>
        </View>
        <View style={styles.headerActions}>
          <Pressable
            accessibilityLabel={t('resetChat')}
            disabled={isSending}
            onPress={() => void resetConversation()}
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
          >
            <Text style={[styles.iconGlyph, { color: colors.onSurface }]}>↺</Text>
          </Pressable>
          <Pressable
            accessibilityLabel={t('settings')}
            onPress={() => navigation.navigate('Settings')}
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
          >
            <Text style={[styles.iconGlyph, { color: colors.onSurface }]}>⚙</Text>
          </Pressable>
        </View>
      </View>

      {messages.length === 0 ? (
        <WelcomeCard
          title={t('welcomeTitle')}
          body={t('welcomeBody')}
          colors={colors}
        />
      ) : (
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContent}
          renderItem={({ item }) => (
            <MessageBubble
              message={item}
              userLabel={t('you')}
              assistantLabel={t('assistant')}
              colors={colors}
            />
          )}
          ItemSeparatorComponent={() => <View style={styles.separator} />}
        />
      )}

      <View
        style={[
          styles.inputBar,
          {
            paddingBottom: Math.max(insets.bottom, 12),
            backgroundColor: colors.background,
          },
        ]}
      >
        <TextInput
          style={[
            styles.input,
            {
              backgroundColor: colors.inputBackground,
              color: colors.onSurface,
            },
          ]}
          placeholder={t('chatHint')}
          placeholderTextColor={colors.outline}
          value={input}
          onChangeText={setInput}
          multiline
          maxLength={4000}
          editable={!isSending}
          onSubmitEditing={() => void handleSend()}
          returnKeyType="send"
        />
        <Pressable
          accessibilityLabel={t('send')}
          disabled={isSending || input.trim().length === 0}
          onPress={() => void handleSend()}
          style={({ pressed }) => [
            styles.sendButton,
            { backgroundColor: colors.primary },
            (isSending || input.trim().length === 0) && styles.sendDisabled,
            pressed && styles.pressed,
          ]}
        >
          {isSending ? (
            <ActivityIndicator color="#FFFFFF" size="small" />
          ) : (
            <Text style={styles.sendGlyph}>➤</Text>
          )}
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  headerText: {
    flex: 1,
    gap: 2,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
  },
  status: {
    fontSize: 12,
  },
  headerActions: {
    flexDirection: 'row',
    gap: 4,
  },
  iconButton: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 20,
  },
  iconGlyph: {
    fontSize: 20,
  },
  listContent: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  separator: {
    height: 12,
  },
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
    paddingHorizontal: 12,
    paddingTop: 8,
  },
  input: {
    flex: 1,
    minHeight: 44,
    maxHeight: 120,
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendDisabled: {
    opacity: 0.5,
  },
  sendGlyph: {
    color: '#FFFFFF',
    fontSize: 16,
  },
  pressed: {
    opacity: 0.7,
  },
});
