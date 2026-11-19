import { useState, type ReactNode } from 'react';
import { Image, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View, type TextInputProps } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { Photo } from '@/contracts/types';
import { palette, radius } from '@/constants/safelog-theme';

export function Page({ title, children }: { title: string; children: ReactNode }) {
  return <SafeAreaView edges={['left', 'right', 'bottom']} style={styles.page}>
    <KeyboardAvoidingView style={styles.page} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.content}>
        <Text accessibilityRole="header" style={styles.title}>{title}</Text>
        {children}
      </ScrollView>
    </KeyboardAvoidingView>
  </SafeAreaView>;
}

export function Card({ children }: { children: ReactNode }) {
  return <View style={styles.card}>{children}</View>;
}

export function Copy({ children, tone = 'normal' }: { children: ReactNode; tone?: 'normal' | 'ai' | 'human' | 'risk' | 'error' }) {
  return <Text accessibilityRole={tone === 'error' ? 'alert' : undefined} style={[styles.copy, {
    color: { normal: palette.ink, ai: palette.blue, human: palette.teal, risk: palette.red, error: palette.red }[tone],
  }]}>{children}</Text>;
}

export function Field({ label, style, ...props }: TextInputProps & { label: string }) {
  return <View style={styles.field}>
    <Text style={styles.label}>{label}</Text>
    <TextInput accessibilityLabel={label} placeholderTextColor={palette.muted} {...props}
      style={[styles.input, props.multiline && styles.multiline, style]} />
  </View>;
}

export function Button({ title, onPress, secondary, disabled }: {
  title: string; onPress: () => void; secondary?: boolean; disabled?: boolean;
}) {
  return <Pressable accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress}
    style={({ pressed }) => [styles.button, secondary && styles.secondary, (pressed || disabled) && styles.dim]}>
    <Text style={[styles.buttonText, secondary && styles.secondaryText]}>{title}</Text>
  </Pressable>;
}

export function PhotoPreview({ photo, label }: { photo?: Photo; label: string }) {
  const [failedUri, setFailedUri] = useState('');
  return <View style={styles.field}>
    <Text style={styles.label}>{label}</Text>
    {photo && photo.uri !== failedUri
      ? <Image accessibilityLabel={label} source={{ uri: photo.uri }} style={styles.photo}
          resizeMode="contain" onError={() => setFailedUri(photo.uri)} />
      : <View style={styles.placeholder}><Copy>{photo ? '사진을 읽을 수 없습니다.' : '사진을 선택하세요.'}</Copy></View>}
  </View>;
}

export const message = (error: unknown) => error instanceof Error ? error.message : '작업을 완료하지 못했습니다.';

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: palette.canvas },
  content: { width: '100%', maxWidth: 620, alignSelf: 'center', padding: 20, gap: 14, paddingBottom: 28 },
  title: { fontSize: 28, fontWeight: '700', color: palette.ink, marginVertical: 4 },
  card: { backgroundColor: palette.surface, borderRadius: radius.card, borderWidth: 1, borderColor: palette.border, padding: 16, gap: 12 },
  copy: { fontSize: 16, lineHeight: 24 },
  field: { gap: 6 },
  label: { color: palette.ink, fontSize: 15, fontWeight: '600' },
  input: { backgroundColor: palette.surface, color: palette.ink, borderWidth: 1, borderColor: palette.border, borderRadius: radius.field, padding: 12, minHeight: 48, fontSize: 16 },
  multiline: { minHeight: 84, textAlignVertical: 'top' },
  button: { minHeight: 48, borderRadius: radius.field, backgroundColor: palette.blue, paddingVertical: 12, paddingHorizontal: 16, justifyContent: 'center', alignItems: 'center' },
  secondary: { backgroundColor: palette.blueSoft },
  dim: { opacity: 0.65 },
  buttonText: { color: palette.surface, fontSize: 16, fontWeight: '600', textAlign: 'center' },
  secondaryText: { color: palette.ink },
  photo: { width: '100%', height: 210, borderRadius: radius.field, backgroundColor: palette.blueSoft },
  placeholder: { minHeight: 100, borderRadius: radius.field, backgroundColor: palette.blueSoft, alignItems: 'center', justifyContent: 'center' },
});
