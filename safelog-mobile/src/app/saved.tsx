import { router } from 'expo-router';
import { Platform } from 'react-native';
import { Button, Card, Copy, Page } from '@/components/safelog-ui';
import { useFinding } from '@/features/workflow/use-finding';
import { StatusBadge } from '@/features/workflow/status-badge';

export default function FindingSavedPage() {
  const { finding } = useFinding();
  return <Page title="점검 등록 완료">
    <Card>{finding ? <>
      <Copy>{finding.location}</Copy>
      <Copy>기록 ID · {finding.id}</Copy>
      <StatusBadge status={finding.status} />
      <Copy>{Platform.OS === 'web'
        ? '개발용 메모리 저장소 · 앱 종료 시 기록이 초기화됩니다.'
        : '기록은 이 기기에 저장됩니다.'}</Copy>
    </> : <Copy>기록을 찾을 수 없습니다.</Copy>}</Card>
    <Button title="홈으로" onPress={() => router.dismissTo('/home')} />
  </Page>;
}
