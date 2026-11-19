import { router, type Href } from 'expo-router';
import { ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { palette, radius } from '@/constants/safelog-theme';
import type { Role } from '@/contracts/types';
import { roleLabels } from '@/features/auth/role-labels';
import { site } from '@/features/auth/demo';
import { useFindings, useSession } from '@/features/auth/session';
import { statusDisplay, workflowStatuses } from '@/features/workflow/status-badge';
import { HomeBackground } from '../components/HomeBackground';
import { HomeHeader } from '../components/HomeHeader';
import { ShortcutCard } from '../components/ShortcutCard';

const taskIcon = require('../../../../assets/safelog/tasks.svg');
const reports = {
  label: '점검 보고서', description: '점검 이력과 보고서 공유', route: '/reports',
  icon: require('../../../../assets/safelog/report.svg'), tint: palette.amberSoft,
} as const;

type Shortcut = { label: string; description: string; icon: number; tint: string; route: Href };

const roleHome: Record<Role, { guidance: string; summary: string; shortcuts: readonly Shortcut[] }> = {
  Inspector: {
    guidance: '현장의 위험을 기록하고 조치 결과를 확인하세요.',
    summary: '내 점검 현황',
    shortcuts: [
      { label: '안전점검 등록', description: '현장 사진과 발견사항 등록', icon: require('../../../../assets/safelog/capture.svg'), tint: palette.blueSoft, route: '/capture' },
      { label: 'AI 위험 분석', description: '사진 분석 및 제안 검토', icon: require('../../../../assets/safelog/scan.svg'), tint: palette.purpleSoft, route: '/analysis' },
      { label: '개선 조치 관리', description: '전후 비교와 최종 확인', icon: taskIcon, tint: palette.tealSoft, route: '/tasks' },
      reports,
    ],
  },
  Manager: {
    guidance: '점검 현황을 확인하고 개선 조치를 배정하세요.',
    summary: '전체 점검 현황',
    shortcuts: [
      { label: '개선 조치 관리', description: '담당자 배정과 진행 상태 확인', icon: taskIcon, tint: palette.tealSoft, route: '/tasks' },
      reports,
    ],
  },
  Assignee: {
    guidance: '배정된 위험 요소를 확인하고 조치 결과를 제출하세요.',
    summary: '내 작업 현황',
    shortcuts: [
      { label: '배정된 작업', description: '내 작업 확인 및 조치 시작', icon: taskIcon, tint: palette.tealSoft, route: '/tasks' },
      { label: '진행 중 작업', description: '조치 사진과 내용 제출', icon: taskIcon, tint: palette.blueSoft, route: { pathname: '/tasks', params: { filter: 'active' } } },
    ],
  },
};

export default function HomeScreen() {
  const { user } = useSession();
  const findings = useFindings();
  const { width, fontScale } = useWindowDimensions();
  if (!user) return null;
  const home = roleHome[user.role];
  const compact = width < 350 || fontScale > 1.3;

  return <SafeAreaView style={styles.page}>
    <ScrollView contentContainerStyle={styles.scroll}>
      <View style={styles.content}>
        <HomeHeader onAccount={() => router.push('/account')} />
        <View style={styles.hero}>
          <HomeBackground />
          <View style={styles.site}><View style={styles.siteDot} /><Text style={styles.siteName}>{site.name}</Text></View>
          <Text accessibilityRole="header" style={styles.greeting}>{user.name}님</Text>
          <Text style={styles.role}>{roleLabels[user.role]}</Text>
          <Text style={styles.guidance}>{home.guidance}</Text>
        </View>
        <View style={styles.section}>
          <Text accessibilityRole="header" style={styles.sectionTitle}>{home.summary}</Text>
          <View style={styles.statusGrid}>
            {workflowStatuses.map(status => {
              const { label, ink, background } = statusDisplay[status];
              const count = findings.filter(finding => finding.status === status).length;
              return <View key={status} accessible accessibilityLabel={label + ' ' + count + '건'}
                style={[styles.status, compact && styles.statusWide, { backgroundColor: background }]}>
                <Text style={[styles.statusCount, { color: ink }]}>{count}</Text>
                <Text style={[styles.statusLabel, { color: ink }]}>{label}</Text>
              </View>;
            })}
          </View>
        </View>
        <View style={styles.section}>
          <Text accessibilityRole="header" style={styles.sectionTitle}>업무 바로가기</Text>
          <View style={styles.shortcuts}>
            {home.shortcuts.map(({ route, ...shortcut }) => <ShortcutCard key={shortcut.label}
              {...shortcut} wide={compact} onPress={() => router.push(route)} />)}
          </View>
        </View>
        <Text style={styles.notice}>시연 모드 · AI Mock · 앱 종료 시 기록 초기화</Text>
      </View>
    </ScrollView>
  </SafeAreaView>;
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: palette.canvas },
  scroll: { alignItems: 'center' },
  content: { width: '100%', maxWidth: 620, paddingHorizontal: 20, paddingTop: 12, paddingBottom: 22, gap: 19 },
  hero: { backgroundColor: palette.navy, overflow: 'hidden', borderRadius: radius.panel, padding: 19, gap: 5 },
  site: { flexDirection: 'row', alignItems: 'center', gap: 7, marginBottom: 5 },
  siteDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: '#73CFE8' },
  siteName: { flexShrink: 1, color: '#C9DBF6', fontWeight: '600', fontSize: 12 },
  greeting: { color: palette.surface, fontWeight: '800', fontSize: 30, letterSpacing: -0.8 },
  role: { alignSelf: 'flex-start', backgroundColor: '#274261', color: '#E4F1FF', borderRadius: radius.pill, fontWeight: '700', paddingVertical: 3, paddingHorizontal: 9, fontSize: 12 },
  guidance: { color: '#E3ECFB', fontSize: 14, lineHeight: 20, marginTop: 6 },
  section: { gap: 9 },
  sectionTitle: { color: palette.ink, fontWeight: '800', fontSize: 18, letterSpacing: -0.35 },
  statusGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  status: { flexBasis: '21%', flexGrow: 1, minHeight: 62, justifyContent: 'center', borderRadius: radius.field, paddingHorizontal: 9, paddingVertical: 7 },
  statusWide: { flexBasis: '44%' },
  statusCount: { fontSize: 22, fontWeight: '800', fontVariant: ['tabular-nums'], includeFontPadding: false },
  statusLabel: { fontSize: 11, fontWeight: '700', marginTop: 3 },
  shortcuts: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  notice: { color: palette.muted, fontSize: 12, textAlign: 'center' },
});
