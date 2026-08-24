export const seedLight = '#3B6EF5';
export const seedDark = '#8AB4FF';

export interface AppColors {
  background: string;
  surface: string;
  surfaceHigh: string;
  surfaceHighest: string;
  primary: string;
  primaryContainer: string;
  onPrimaryContainer: string;
  onSurface: string;
  onSurfaceVariant: string;
  outline: string;
  errorContainer: string;
  onErrorContainer: string;
  inputBackground: string;
  border: string;
}

export function lightColors(): AppColors {
  return {
    background: '#F8F9FC',
    surface: '#F8F9FC',
    surfaceHigh: '#E8ECF4',
    surfaceHighest: '#E1E6F0',
    primary: seedLight,
    primaryContainer: '#D6E2FF',
    onPrimaryContainer: '#0A1F5C',
    onSurface: '#1A1C20',
    onSurfaceVariant: '#44474F',
    outline: '#74777F',
    errorContainer: '#FFDAD6',
    onErrorContainer: '#410002',
    inputBackground: 'rgba(225, 230, 240, 0.5)',
    border: '#D0D5E0',
  };
}

export function darkColors(): AppColors {
  return {
    background: '#111318',
    surface: '#111318',
    surfaceHigh: '#1E2128',
    surfaceHighest: '#282B33',
    primary: seedDark,
    primaryContainer: '#1F3A7A',
    onPrimaryContainer: '#D6E2FF',
    onSurface: '#E2E2E6',
    onSurfaceVariant: '#C4C6D0',
    outline: '#8E9099',
    errorContainer: '#93000A',
    onErrorContainer: '#FFDAD6',
    inputBackground: 'rgba(40, 43, 51, 0.5)',
    border: '#3A3D47',
  };
}
