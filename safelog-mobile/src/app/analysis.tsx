import { router } from 'expo-router';
import { Button, Card, Copy, Page } from '@/components/safelog-ui';
import { useFindings, useSession } from '@/features/auth/session';
import { StatusBadge } from '@/features/workflow/status-badge';
import type { Decision } from '@/contracts/types';

const decisionLabels: Record<Decision, string> = {
  Pending: 'AI 제안 검토 대기',
  Accepted: 'AI 제안 채택',
  Edited: 'AI 제안 수정 후 채택',
  Rejected: 'AI 제안 거절',
};

export default function AnalysisQueuePage() {
  const { user } = useSession();
  const findings = useFindings();
  if (user?.role !== 'Inspector') return null;
  const open = findings.filter(finding => finding.status === 'Open');

  return <Page title="AI 위험 분석">
    <Copy tone="ai">시연용 AI 제안을 확인하고 점검자가 채택 여부를 결정합니다.</Copy>
    {open.length === 0 && <Card>
      <Copy>{findings.length ? '현재 분석하거나 검토할 미조치 기록이 없습니다.' : '분석할 사진이 아직 없습니다. 점검을 등록해 사진을 추가하세요.'}</Copy>
      <Button title="안전점검 등록" onPress={() => router.push('/capture')} />
    </Card>}
    {open.map(finding => {
      const latest = finding.analyses.filter(item => item.type === 'BeforeHazard').at(-1);
      const lastReview = finding.logs
        .filter(log => log.action === 'hazard_reviewed')
        .at(-1);
      const wasReviewed = Boolean(lastReview);
      const manuallyReviewed = lastReview?.note === 'Manual'
        && latest?.decision !== 'Pending';
      return <Card key={finding.id}>
        <Copy>{finding.location}</Copy>
        <StatusBadge status={finding.status} />
        <Copy>
          {manuallyReviewed
            ? '수동 검토 완료'
            : latest
              ? decisionLabels[latest.decision]
              : wasReviewed
                ? '수동 검토 완료'
                : 'AI 분석 전'}
        </Copy>
        <Button title={latest?.decision === 'Pending' ? 'AI 제안 검토' : latest ? '분석 화면 열기' : 'AI 분석 시작'}
          onPress={() => router.push({
            pathname: '/hazard', params: { id: finding.id, ...(latest ? {} : { fresh: '1' }) },
          })} />
      </Card>;
    })}
  </Page>;
}
