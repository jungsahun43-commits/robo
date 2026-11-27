import { useState } from 'react';
import { router } from 'expo-router';
import { Switch } from 'react-native';
import { Button, Card, Copy, Field, message, Page, PhotoPreview } from '@/components/safelog-ui';
import { AnalysisPanel } from '@/features/ai/analysis-panel';
import type { MockResponse } from '@/features/ai/mock';
import { useAnalysis } from '@/features/ai/use-analysis';
import { workflow } from '@/features/auth/session';
import { ownsFinding } from '@/features/workflow/service';
import { StatusBadge } from '@/features/workflow/status-badge';
import { useFinding } from '@/features/workflow/use-finding';

export default function AiComparisonReviewPage() {
  const { user, finding } = useFinding();
  const analysis = useAnalysis(`${user?.id}/${finding?.id}/${finding?.aiMode}/${finding?.inspectionTask}/${finding?.after.length}/${finding?.after.at(-1)?.uri}`,
    finding?.aiMode, finding?.inspectionTask);
  const [note, setNote] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [decision, setDecision] = useState<'Accepted' | 'Rejected'>('Rejected');
  const [manual, setManual] = useState(finding?.aiMode === 'manual');
  const [error, setError] = useState('');
  const after = finding?.after.at(-1);
  if (!finding || !user || !after) return <Page title="조치 전후 비교"><Copy>비교 가능한 작업이 없습니다.</Copy></Page>;
  const current = finding;
  const actor = user;
  const afterPhoto = after;
  const mayReview = ownsFinding(user, finding) && finding.status === 'PendingReview';
  const result = finding.analyses.filter(item =>
    item.type === 'AfterComparison' && item.submissionCount === finding.after.length).at(-1);

  function run(mode: MockResponse) {
    setManual(false);
    setConfirmed(false);
    setDecision('Rejected');
    void analysis.run(async (analyzer, signal) => {
      const suggestion = await analyzer.compare(current.photo, afterPhoto, signal);
      if (!analysis.isCurrent(signal)) throw new Error('분석 취소');
      const result = workflow.execute(actor, current.id, { kind: 'suggest', type: 'AfterComparison', suggestion,
        expectedSubmissionCount: current.after.length, expectedAfterUri: afterPhoto.uri });
      if (!('type' in result)) throw new Error('분석 기록을 저장할 수 없습니다.');
      return result;
    }, mode);
  }

  function finalize(verify: boolean) {
    try {
      setError('');
      if (verify) {
        workflow.execute(actor, current.id, {
          kind: 'verify', note, confirmed,
          analysisId: !analysis.error && !manual && result?.decision === 'Pending' ? result.id : undefined,
          decision: manual ? 'Rejected' : decision,
        });
      } else {
        workflow.execute(actor, current.id, { kind: 'changes', note });
        router.dismissTo('/tasks');
      }
    } catch (error) {
      setError(message(error));
    }
  }

  return <Page title="조치 전후 비교">
    <Card>
      <Copy>{finding.location}</Copy>
      <StatusBadge status={finding.status} />
      <PhotoPreview photo={finding.photo} label="조치 전 사진" />
      <PhotoPreview photo={after} label="조치 후 사진" />
      <Copy>{finding.logs.filter(log => log.action === 'action_submitted').at(-1)?.note}</Copy>
    </Card>
    {finding.status === 'PendingReview' && <AnalysisPanel {...analysis} aiMode={finding.aiMode} inspectionTask={finding.inspectionTask}
      comparison result={manual ? undefined : result} onRun={run} onManual={() => {
        analysis.cancel();
        setManual(true);
        setConfirmed(false);
      }} />}
    {mayReview ? <>
      <Card>
        <Copy tone="human">사람의 최종 판단</Copy>
        {!manual && !analysis.error && !analysis.busy && result?.decision === 'Pending' && <Button secondary title={'AI 참고 제안 검토 · ' + decision}
          onPress={() => setDecision(decision === 'Accepted' ? 'Rejected' : 'Accepted')} />}
        <Field label="확인 내용 또는 추가 조치 요청 사유" value={note} onChangeText={setNote} multiline />
        <Copy>전후 사진과 조치 내용을 직접 확인했습니다.</Copy>
        <Switch accessibilityLabel="전후 사진과 조치 내용을 직접 확인했습니다" value={confirmed} onValueChange={setConfirmed} />
        <Button title="최종 확인" disabled={analysis.busy || !confirmed || !note.trim()} onPress={() => finalize(true)} />
        <Button title="추가 조치 요청" secondary disabled={analysis.busy} onPress={() => finalize(false)} />
      </Card>
    </> : <>
      {finding.status === 'Verified' && result && <Copy tone="ai">AI 참고 제안 · {result.assessment} ({result.decision})</Copy>}
      {finding.status === 'Verified' ? <Copy tone="human">점검자 최종 확인 완료</Copy>
        : <Copy>원 점검자가 최종 확인해야 합니다.</Copy>}
    </>}
    {finding.status === 'Verified' && user.role !== 'Assignee' && <Button title="보고서 보기" onPress={() => router.push({ pathname: '/report', params: { id: finding.id } })} />}
    {!!error && <Copy tone="error">{error}</Copy>}
  </Page>;
}
