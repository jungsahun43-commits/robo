import { StyleSheet, Text, View } from 'react-native';
import { palette, radius } from '@/constants/safelog-theme';
import type { Status } from '@/contracts/types';

export const statusDisplay = {
  Open: { label: '미조치', ink: palette.red, background: palette.redSoft },
  InProgress: { label: '조치 중', ink: palette.blue, background: palette.blueSoft },
  PendingReview: { label: '검토 대기', ink: palette.amber, background: palette.amberSoft },
  Verified: { label: '확인 완료', ink: palette.teal, background: palette.tealSoft },
} satisfies Record<Status, { label: string; ink: string; background: string }>;

export const workflowStatuses = Object.keys(statusDisplay) as Status[];

export function StatusBadge({ status }: { status: Status }) {
  const { label, ink, background } = statusDisplay[status];
  return <View style={[styles.badge, { backgroundColor: background }]}>
    <View style={[styles.dot, { backgroundColor: ink }]} />
    <Text style={[styles.label, { color: ink }]}>{label}</Text>
  </View>;
}

const styles = StyleSheet.create({
  badge: { alignSelf: 'flex-start', flexDirection: 'row', alignItems: 'center', gap: 5, borderRadius: radius.pill, paddingVertical: 4, paddingHorizontal: 9 },
  dot: { width: 6, height: 6, borderRadius: 3 },
  label: { fontSize: 12, fontWeight: '700' },
});
