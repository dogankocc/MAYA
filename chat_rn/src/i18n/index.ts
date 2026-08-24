import * as Localization from 'expo-localization';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import en from './locales/en.json';
import tr from './locales/tr.json';

function deviceLocale(): 'tr' | 'en' {
  const code = Localization.getLocales()[0]?.languageCode ?? 'en';
  return code.startsWith('tr') ? 'tr' : 'en';
}

void i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    tr: { translation: tr },
  },
  lng: deviceLocale(),
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
});

export function applyLocalePreference(locale: 'tr' | 'en' | null): void {
  void i18n.changeLanguage(locale ?? deviceLocale());
}

export default i18n;
