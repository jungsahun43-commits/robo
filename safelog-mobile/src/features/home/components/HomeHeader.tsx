import { Image } from 'expo-image';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { palette, radius } from '@/constants/safelog-theme';

export function HomeHeader({ onAccount }: { onAccount: () => void }) {
  return <View style={styles.header}>
    <Image accessible={false} source={require('../../../../assets/safelog/brand.svg')} style={styles.logo} />
    <View style={styles.brand}>
      <Text accessibilityRole="header" style={styles.name}>SafeLog <Text style={styles.ai}>AI</Text></Text>
      <Text style={styles.subtitle}>산업안전 점검</Text>
    </View>
    <Pressable accessibilityRole="button" accessibilityLabel="내 계정" onPress={onAccount}
      style={({ pressed }) => [styles.account, pressed && styles.pressed]}>
      <Image accessible={false} source={require('../../../../assets/safelog/account.svg')} style={styles.accountIcon} />
    </Pressable>
  </View>;
}

const styles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  logo: { width: 43, height: 43 },
  brand: { flex: 1 },
  name: { color: palette.ink, fontSize: 24, fontWeight: '800', letterSpacing: -0.7, includeFontPadding: false },
  ai: { color: palette.blue },
  subtitle: { color: palette.muted, fontSize: 12, marginTop: 2 },
  account: { width: 48, height: 48, alignItems: 'center', justifyContent: 'center', borderRadius: radius.card, backgroundColor: palette.surface, borderWidth: 1, borderColor: palette.border },
  accountIcon: { width: 24, height: 24 },
  pressed: { opacity: 0.68 },
});
