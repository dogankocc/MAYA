import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { useColorScheme } from 'react-native';

import { ChatApi, ChatApiException } from '@/api/chatApi';
import { applyLocalePreference } from '@/i18n';
import type { ChatMessage } from '@/models/chatMessage';
import {
  defaultSettings,
  SettingsService,
  type AppSettings,
} from '@/services/settingsService';
import { darkColors, lightColors, type AppColors } from '@/theme/colors';

export type ServerConnectionState = 'unknown' | 'connected' | 'disconnected';

interface ChatContextValue {
  settings: AppSettings;
  messages: ChatMessage[];
  connectionState: ServerConnectionState;
  serverStage: string | null;
  serverModel: string | null;
  isSending: boolean;
  colors: AppColors;
  isDark: boolean;
  initialized: boolean;
  updateSettings: (settings: AppSettings) => Promise<ServerConnectionState>;
  refreshHealth: (serverUrl?: string) => Promise<ServerConnectionState>;
  resetConversation: () => Promise<void>;
  sendMessage: (text: string) => Promise<void>;
}

const ChatContext = createContext<ChatContextValue | null>(null);

const settingsService = new SettingsService();

export function ChatProvider({ children }: { children: ReactNode }) {
  const systemScheme = useColorScheme();
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [connectionState, setConnectionState] =
    useState<ServerConnectionState>('unknown');
  const [serverStage, setServerStage] = useState<string | null>(null);
  const [serverModel, setServerModel] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [messageCounter, setMessageCounter] = useState(0);

  const isDark =
    settings.themeMode === 'dark' ||
    (settings.themeMode === 'system' && systemScheme === 'dark');

  const colors = useMemo(() => (isDark ? darkColors() : lightColors()), [isDark]);

  const applyBackendConfig = useCallback(async (serverUrl: string, next: AppSettings) => {
    const config = await new ChatApi(serverUrl).setConfig({
      backend: next.modelBackend,
      openaiModel: next.openaiModel,
    });
    setServerModel(config.model);
    setServerStage(config.backend === 'local' ? 'local-mini' : 'ollama-compatible');
  }, []);

  const refreshHealth = useCallback(async (serverUrl = settings.serverUrl): Promise<ServerConnectionState> => {
    setConnectionState('unknown');
    try {
      const health = await new ChatApi(serverUrl).health();
      setServerStage(health.stage);
      setServerModel(health.model ?? null);
      const nextState: ServerConnectionState = health.loaded ? 'connected' : 'disconnected';
      setConnectionState(nextState);
      return nextState;
    } catch {
      setServerStage(null);
      setServerModel(null);
      setConnectionState('disconnected');
      return 'disconnected';
    }
  }, [settings.serverUrl]);

  const updateSettings = useCallback(
    async (next: AppSettings): Promise<ServerConnectionState> => {
      const backendChanged =
        next.modelBackend !== settings.modelBackend ||
        next.openaiModel !== settings.openaiModel ||
        next.serverUrl !== settings.serverUrl;

      setSettings(next);
      await settingsService.save(next);
      applyLocalePreference(next.locale);

      try {
        await applyBackendConfig(next.serverUrl, next);
        if (backendChanged) {
          setMessages([]);
        }
      } catch {
        return 'disconnected';
      }

      return refreshHealth(next.serverUrl);
    },
    [applyBackendConfig, refreshHealth, settings.modelBackend, settings.openaiModel, settings.serverUrl],
  );

  const resetConversation = useCallback(async () => {
    await new ChatApi(settings.serverUrl).reset();
    setMessages([]);
  }, [settings.serverUrl]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isSending) {
        return;
      }

      setIsSending(true);
      let counter = messageCounter;

      counter += 1;
      const userId = `u${counter}`;
      setMessages((prev) => [
        ...prev,
        {
          id: userId,
          role: 'user',
          text: trimmed,
          timestamp: new Date(),
        },
      ]);
      setMessageCounter(counter);

      try {
        const reply = await new ChatApi(settings.serverUrl).complete({
          prompt: trimmed,
          maxTokens: settings.maxTokens,
          temperature: settings.temperature,
          greedy: settings.greedy,
        });

        counter += 1;
        setMessages((prev) => [
          ...prev,
          {
            id: `a${counter}`,
            role: 'assistant',
            text: reply.text.length > 0 ? reply.text : '...',
            timestamp: new Date(),
          },
        ]);
        setMessageCounter(counter);
        setConnectionState('connected');
      } catch (error) {
        counter += 1;
        const message =
          error instanceof ChatApiException
            ? error.message
            : error instanceof Error
              ? error.message
              : String(error);

        setMessages((prev) => [
          ...prev,
          {
            id: `e${counter}`,
            role: 'assistant',
            text: message,
            timestamp: new Date(),
            isError: true,
          },
        ]);
        setMessageCounter(counter);
        setConnectionState('disconnected');
      } finally {
        setIsSending(false);
      }
    },
    [isSending, messageCounter, settings],
  );

  useEffect(() => {
    let active = true;

    void (async () => {
      const loaded = await settingsService.load();
      if (!active) {
        return;
      }
      setSettings(loaded);
      applyLocalePreference(loaded.locale);
      setInitialized(true);
      try {
        await applyBackendConfig(loaded.serverUrl, loaded);
      } catch {
        // server may be offline during startup
      }
      await refreshHealth(loaded.serverUrl);
    })();

    return () => {
      active = false;
    };
  }, [applyBackendConfig, refreshHealth]);

  const value = useMemo(
    () => ({
      settings,
      messages,
      connectionState,
      serverStage,
      serverModel,
      isSending,
      colors,
      isDark,
      initialized,
      updateSettings,
      refreshHealth,
      resetConversation,
      sendMessage,
    }),
    [
      settings,
      messages,
      connectionState,
      serverStage,
      serverModel,
      isSending,
      colors,
      isDark,
      initialized,
      updateSettings,
      refreshHealth,
      resetConversation,
      sendMessage,
    ],
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatContextValue {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within ChatProvider');
  }
  return context;
}
