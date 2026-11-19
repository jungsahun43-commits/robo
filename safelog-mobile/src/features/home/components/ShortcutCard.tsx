import { Image, type ImageSource } from 'expo-image';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { palette, radius } from '@/constants/safelog-theme';

type Props = {
  label: string;
  description: string;
  icon: ImageSource | number;
  tint: string;
  wide: boolean;
  onPress: () => void;
};

export function ShortcutCard({ label, description, icon, tint, wide, onPress }: Props) {
  return <Pressable accessibilityRole="button" accessibilityLabel={label}
    accessibilityHint={description} onPress={onPress}
    style={({ pressed }) => [styles.card, wide && styles.wide, pressed && styles.pressed]}>
    <View style={styles.top}>
      <View style={[styles.iconTile, { backgroundColor: tint }]}>
        <Image accessible={false} source={icon} style={styles.icon} />
      </View>
      <Text accessible={false} importantForAccessibility="no" style={styles.arrow}>↗</Text>
    </View>
    <Text style={styles.label}>{label}</Text>
    <Text style={styles.description}>{description}</Text>
  </Pressable>;
}

const styles = StyleSheet.create({
  card: { flexBasis: '46%', flexGrow: 1, minHeight: 132, padding: 14, borderRadius: radius.card, borderWidth: 1, borderColor: palette.border, backgroundColor: palette.surface },
  wide: { flexBasis: '100%' },
  top: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 },
  iconTile: { height: 36, width: 36, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  icon: { width: 22, height: 22 },
  arrow: { color: palette.muted, fontSize: 17, lineHeight: 20 },
  label: { color: palette.ink, fontWeight: '800', fontSize: 16, marginBottom: 3 },
  description: { color: palette.muted, fontSize: 12, lineHeight: 17 },
  pressed: { backgroundColor: palette.blueSoft, borderColor: palette.blue },
});
