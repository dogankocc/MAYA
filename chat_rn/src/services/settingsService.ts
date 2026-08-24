import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

export type ThemeMode = 'system' | 'light' | 'dark';
export type ModelBackend = 'local' | 'openai';

export const OPENAI_MODEL_OPTIONS = ['llama3.2', 'mistral', 'gemma2'] as const;
export type OpenAiModelOption = (typeof OPENAI_MODEL_OPTIONS)[number];

export interface AppSettings {
  serverUrl: string;
  maxTokens: number;
  temperature: number;
  greedy: boolean;
  themeMode: ThemeMode;
  locale: 'tr' | 'en' | null;
  modelBackend: ModelBackend;
  openaiModel: OpenAiModelOption;
}

const STORAGE_KEYS = {
  serverUrl: 'server_url',
  maxTokens: 'max_tokens',
  temperature: 'temperature',
  greedy: 'greedy',
  themeMode: 'theme_mode',
  locale: 'locale',
  modelBackend: 'model_backend',
  openaiModel: 'openai_model',
} as const;

export function defaultServerUrl(): string {
  return Platform.OS === 'android' ? 'http://10.0.2.2:8765' : 'http://127.0.0.1:8765';
}

export const defaultSettings: AppSettings = {
  serverUrl: defaultServerUrl(),
  maxTokens: 256,
  temperature: 0.7,
  greedy: false,
  themeMode: 'system',
  locale: null,
  modelBackend: 'openai',
  openaiModel: 'llama3.2',
};

export class SettingsService {
  async load(): Promise<AppSettings> {
    const entries = await AsyncStorage.multiGet(Object.values(STORAGE_KEYS));
    const values = Object.fromEntries(entries) as Record<string, string | null>;

    const themeMode = values[STORAGE_KEYS.themeMode];
    const locale = values[STORAGE_KEYS.locale];
    const modelBackend = values[STORAGE_KEYS.modelBackend];
    const openaiModel = values[STORAGE_KEYS.openaiModel];

    return {
      serverUrl: values[STORAGE_KEYS.serverUrl] ?? defaultServerUrl(),
      maxTokens: values[STORAGE_KEYS.maxTokens]
        ? Number.parseInt(values[STORAGE_KEYS.maxTokens]!, 10)
        : defaultSettings.maxTokens,
      temperature: values[STORAGE_KEYS.temperature]
        ? Number.parseFloat(values[STORAGE_KEYS.temperature]!)
        : defaultSettings.temperature,
      greedy: values[STORAGE_KEYS.greedy] === 'true',
      themeMode:
        themeMode === 'light' || themeMode === 'dark' || themeMode === 'system'
          ? themeMode
          : defaultSettings.themeMode,
      locale: locale === 'tr' || locale === 'en' ? locale : null,
      modelBackend: modelBackend === 'local' ? 'local' : defaultSettings.modelBackend,
      openaiModel: OPENAI_MODEL_OPTIONS.includes(openaiModel as OpenAiModelOption)
        ? (openaiModel as OpenAiModelOption)
        : defaultSettings.openaiModel,
    };
  }

  async save(settings: AppSettings): Promise<void> {
    const pairs: [string, string][] = [
      [STORAGE_KEYS.serverUrl, settings.serverUrl],
      [STORAGE_KEYS.maxTokens, String(settings.maxTokens)],
      [STORAGE_KEYS.temperature, String(settings.temperature)],
      [STORAGE_KEYS.greedy, String(settings.greedy)],
      [STORAGE_KEYS.themeMode, settings.themeMode],
      [STORAGE_KEYS.modelBackend, settings.modelBackend],
      [STORAGE_KEYS.openaiModel, settings.openaiModel],
    ];

    if (settings.locale == null) {
      await AsyncStorage.removeItem(STORAGE_KEYS.locale);
    } else {
      pairs.push([STORAGE_KEYS.locale, settings.locale]);
    }

    await AsyncStorage.multiSet(pairs);
  }
}
