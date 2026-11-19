import { router } from 'expo-router';
import { Button, Card, Copy, Page } from '@/components/safelog-ui';
import { useFindings, useSession } from '@/features/auth/session';
import { StatusBadge } from '@/features/workflow/status-badge';

export default function SelectReportPage() {
  const { user } = useSession();
  const records = useFindings();
  if (!user || user.role === 'Assignee') return null;

  return <Page title="점검 보고서">
    {!records.length && <Card><Copy>보고서를 작성할 점검 기록이 없습니다.</Copy></Card>}
    {records.map(finding => <Card key={finding.id}>
      <Copy>{finding.location}</Copy>
      <StatusBadge status={finding.status} />
      <Button title="보고서 미리보기" onPress={() => router.push({
        pathname: '/report', params: { id: finding.id },
      })} />
    </Card>)}
  </Page>;
}
