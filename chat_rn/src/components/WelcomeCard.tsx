import { StyleSheet, Text, View } from 'react-native';

import type { AppColors } from '@/theme/colors';

interface WelcomeCardProps {
  title: string;
  body: string;
  colors: AppColors;
}

export function WelcomeCard({ title, body, colors }: WelcomeCardProps) {
  return (
    <View style={styles.wrapper}>
      <View style={[styles.card, { backgroundColor: colors.surfaceHighest }]}>
        <Text style={[styles.icon, { color: colors.primary }]}>✦</Text>
        <Text style={[styles.title, { color: colors.onSurface }]}>{title}</Text>
        <Text style={[styles.body, { color: colors.onSurfaceVariant }]}>{body}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
  },
  card: {
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    gap: 8,
  },
  icon: {
    fontSize: 40,
    marginBottom: 4,
  },
  title: {
    fontSize: 22,
    fontWeight: '600',
  },
  body: {
    fontSize: 15,
    textAlign: 'center',
    lineHeight: 22,
  },
});
