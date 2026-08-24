import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import { useChat } from '@/context/ChatContext';
import type { RootStackParamList } from '@/navigation/types';
import {
  defaultServerUrl,
  OPENAI_MODEL_OPTIONS,
  type AppSettings,
  type ModelBackend,
  type ThemeMode,
} from '@/services/settingsService';

type Props = NativeStackScreenProps<RootStackParamList, 'Settings'>;

type LocaleOption = AppSettings['locale'];

function SegmentButton<T extends string>({
  value,
  selected,
  label,
  onPress,
  colors,
}: {
  value: T;
  selected: boolean;
  label: string;
  onPress: (value: T) => void;
  colors: ReturnType<typeof useChat>['colors'];
}) {
  return (
    <Pressable
      onPress={() => onPress(value)}
      style={[
        styles.segment,
        {
          backgroundColor: selected ? colors.primaryContainer : colors.surfaceHigh,
          borderColor: colors.border,
        },
      ]}
    >
      <Text
        style={{
          color: selected ? colors.onPrimaryContainer : colors.onSurface,
          fontWeight: selected ? '600' : '400',
        }}
      >
        {label}
      </Text>
    </Pressable>
  );
}

export function SettingsScreen({ navigation }: Props) {
  const { t } = useTranslation();
  const insets = useSafeAreaInsets();
  const { settings, colors, updateSettings } = useChat();

  const [serverUrl, setServerUrl] = useState(settings.serverUrl);
  const [maxTokens, setMaxTokens] = useState(settings.maxTokens);
  const [temperature, setTemperature] = useState(settings.temperature);
  const [greedy, setGreedy] = useState(settings.greedy);
  const [themeMode, setThemeMode] = useState<ThemeMode>(settings.themeMode);
  const [locale, setLocale] = useState<LocaleOption>(settings.locale);
  const [modelBackend, setModelBackend] = useState<ModelBackend>(settings.modelBackend);
  const [openaiModel, setOpenaiModel] = useState(settings.openaiModel);
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);

  const buildSettings = (): AppSettings => ({
    serverUrl: serverUrl.trim(),
    maxTokens,
    temperature,
    greedy,
    themeMode,
    locale,
    modelBackend,
    openaiModel,
  });

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    const state = await updateSettings(buildSettings());
    setTestResult(state === 'connected' ? t('connected') : t('errorHealth'));
    setTesting(false);
  };

  const handleSave = async () => {
    setSaving(true);
    await updateSettings(buildSettings());
    setSaving(false);
    navigation.goBack();
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={[
        styles.content,
        { paddingBottom: Math.max(insets.bottom, 24) },
      ]}
    >
      <Text style={[styles.label, { color: colors.onSurfaceVariant }]}>
        {t('serverUrl')}
      </Text>
      <TextInput
        style={[
          styles.field,
          {
            backgroundColor: colors.inputBackground,
            color: colors.onSurface,
            borderColor: colors.border,
          },
        ]}
        value={serverUrl}
        onChangeText={setServerUrl}
        placeholder={defaultServerUrl()}
        placeholderTextColor={colors.outline}
        autoCapitalize="none"
        autoCorrect={false}
      />

      <Pressable
        onPress={() => void handleTest()}
        disabled={testing}
        style={[
          styles.outlineButton,
          { borderColor: colors.primary },
          testing && styles.disabled,
        ]}
      >
        {testing ? (
          <ActivityIndicator color={colors.primary} />
        ) : (
          <Text style={{ color: colors.primary, fontWeight: '600' }}>
            {t('testConnection')}
          </Text>
        )}
      </Pressable>
      {testResult ? (
        <Text style={[styles.hint, { color: colors.onSurfaceVariant }]}>
          {testResult}
        </Text>
      ) : null}

      <Text style={[styles.sectionTitle, { color: colors.onSurface }]}>
        {t('modelSource')}
      </Text>
      <View style={styles.segmentRow}>
        <SegmentButton
          value="local"
          selected={modelBackend === 'local'}
          label={t('modelLocal')}
          onPress={() => setModelBackend('local')}
          colors={colors}
        />
        <SegmentButton
          value="openai"
          selected={modelBackend === 'openai'}
          label={t('modelOpenAi')}
          onPress={() => setModelBackend('openai')}
          colors={colors}
        />
      </View>

      {modelBackend === 'openai' ? (
        <>
          <Text style={[styles.sectionTitle, { color: colors.onSurface }]}>
            {t('openAiModel')}
          </Text>
          <View style={styles.segmentRow}>
            {OPENAI_MODEL_OPTIONS.map((model) => (
              <SegmentButton
                key={model}
                value={model}
                selected={openaiModel === model}
                label={model}
                onPress={setOpenaiModel}
                colors={colors}
              />
            ))}
          </View>
        </>
      ) : null}

      <Text style={[styles.sectionTitle, { color: colors.onSurface }]}>
        {t('maxTokens')}: {maxTokens}
      </Text>
      <View style={styles.sliderRow}>
        {[8, 32, 64, 128, 256].map((value) => (
          <SegmentButton
            key={value}
            value={String(value)}
            selected={maxTokens === value}
            label={String(value)}
            onPress={(v) => setMaxTokens(Number.parseInt(v, 10))}
            colors={colors}
          />
        ))}
      </View>

      <Text style={[styles.sectionTitle, { color: colors.onSurface }]}>
        {t('temperature')}: {temperature.toFixed(1)}
      </Text>
      <View style={styles.sliderRow}>
        {[0.1, 0.5, 0.8, 1.0, 1.5].map((value) => (
          <SegmentButton
            key={value}
            value={String(value)}
            selected={Math.abs(temperature - value) < 0.01}
            label={value.toFixed(1)}
            onPress={(v) => setTemperature(Number.parseFloat(v))}
            colors={colors}
          />
        ))}
      </View>

      <View style={styles.switchRow}>
        <Text style={[styles.sectionTitle, { color: colors.onSurface, flex: 1 }]}>
          {t('greedyMode')}
        </Text>
        <Switch value={greedy} onValueChange={setGreedy} />
      </View>

      <Text style={[styles.sectionTitle, { color: colors.onSurface }]}>
        {t('theme')}
      </Text>
      <View style={styles.segmentRow}>
        {(['system', 'light', 'dark'] as ThemeMode[]).map((mode) => (
          <SegmentButton
            key={mode}
            value={mode}
            selected={themeMode === mode}
            label={t(
              mode === 'system'
                ? 'themeSystem'
                : mode === 'light'
                  ? 'themeLight'
                  : 'themeDark',
            )}
            onPress={setThemeMode}
            colors={colors}
          />
        ))}
      </View>

      <Text style={[styles.sectionTitle, { color: colors.onSurface }]}>
        {t('language')}
      </Text>
      <View style={styles.segmentRow}>
        <SegmentButton
          value="auto"
          selected={locale === null}
          label={t('languageAuto')}
          onPress={() => setLocale(null)}
          colors={colors}
        />
        <SegmentButton
          value="tr"
          selected={locale === 'tr'}
          label="TR"
          onPress={() => setLocale('tr')}
          colors={colors}
        />
        <SegmentButton
          value="en"
          selected={locale === 'en'}
          label="EN"
          onPress={() => setLocale('en')}
          colors={colors}
        />
      </View>

      <Pressable
        onPress={() => void handleSave()}
        disabled={saving}
        style={[
          styles.saveButton,
          { backgroundColor: colors.primary },
          saving && styles.disabled,
        ]}
      >
        {saving ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          <Text style={styles.saveText}>{t('save')}</Text>
        )}
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 16,
    gap: 12,
  },
  label: {
    fontSize: 14,
  },
  field: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
  },
  outlineButton: {
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
  },
  hint: {
    fontSize: 13,
  },
  sectionTitle: {
    fontSize: 15,
    fontWeight: '600',
    marginTop: 8,
  },
  segmentRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  sliderRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  segment: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  saveButton: {
    marginTop: 16,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  saveText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 16,
  },
  disabled: {
    opacity: 0.6,
  },
});
