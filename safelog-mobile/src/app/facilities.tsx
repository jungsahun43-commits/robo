import { router } from 'expo-router';
import { Button, Card, Copy, Page } from '@/components/safelog-ui';
import { site } from '@/features/auth/demo';
import { useFindings, useSession } from '@/features/auth/session';

export default function FacilityOverviewPage() {
  const { user } = useSession();
  const findings = useFindings();
  if (!user) return null;
  const count = findings.filter(finding => finding.siteId === site.id).length;

  return <Page title="시설물현황">
    <Card>
      <Copy>{site.name}</Copy>
      <Copy>{site.address}</Copy>
      <Copy>조회 가능한 점검 기록 {count}건</Copy>
    </Card>
    <Copy>시연 현장 데이터입니다. 시설물 대장 연동은 준비 중입니다.</Copy>
    <Button title="점검관리 열기" onPress={() => router.push('/tasks')} />
  </Page>;
}
