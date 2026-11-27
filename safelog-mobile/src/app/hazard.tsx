import { useEffect, useState } from 'react';
import { router, useLocalSearchParams } from 'expo-router';
import { Button, Card, Copy, Field, message, Page, PhotoPreview } from '@/components/safelog-ui';
import { AnalysisPanel } from '@/features/ai/analysis-panel';
import type { MockResponse } from '@/features/ai/mock';
import { useAnalysis } from '@/features/ai/use-analysis';
import { workflow } from '@/features/auth/session';
import { useFinding } from '@/features/workflow/use-finding';

export default function AiHazardReviewPage() {
  const { finding, user } = useFinding();
  const { fresh } = useLocalSearchParams<{ fresh?: string }>();
  const analysis = useAnalysis(`${user?.id}/${finding?.id}/${finding?.aiMode}/${finding?.inspectionTask}`,
    finding?.aiMode, finding?.inspectionTask);
  const latest = finding?.analyses.filter(item => item.type === 'BeforeHazard').at(-1);
  const pending = latest?.decision === 'Pending' ? latest : undefined;
  const [manual, setManual] = useState(finding?.aiMode === 'manual');
  const [description, setDescription] = useState(pending?.assessment ?? finding?.description ?? '');
  const [action, setAction] = useState(pending?.action ?? finding?.action ?? '');
  const [error, setError] = useState('');

  function run(mode: MockResponse) {
    if (!finding || !user) return;
    setManual(false);
    void analysis.run(async (analyzer, signal) => {
      const suggestion = await analyzer.hazard(finding.photo, signal);
      if (!analysis.isCurrent(signal)) throw new Error('분석 취소');
      const result = workflow.execute(user, finding.id, { kind: 'suggest', type: 'BeforeHazard', suggestion });
      if (!('type' in result)) throw new Error('분석 기록을 저장할 수 없습니다.');
      setDescription(result.assessment);
      setAction(result.action);
      return result;
    }, mode);
  }

  useEffect(() => {
    if (fresh !== '1' || finding?.aiMode === 'manual') return;
    const timer = setTimeout(() => run('정상'), 0);
    return () => clearTimeout(timer);
    // Start once for a newly registered finding.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!finding || !user || finding.status !== 'Open') {
    return <Page title="AI 위험 검토"><Copy>검토 가능한 기록이 없습니다.</Copy></Page>;
  }
  const result = analysis.result ?? pending;
  const successful = !analysis.busy && !analysis.error && result;

  function save(reject: boolean) {
    if (!finding || !user) return;
    try {
      const decision = reject || manual || !result ? 'Rejected'
        : description.trim() === result.assessment && action.trim() === result.action ? 'Accepted' : 'Edited';
      workflow.execute(user, finding.id, {
        kind: 'hazard', analysisId: result?.id, decision,
        description: reject ? finding.description : description, action: reject ? finding.action : action,
      });
      router.replace({ pathname: '/saved', params: { id: finding.id } });
    } catch (error) {
      setError(message(error));
    }
  }

  return <Page title="AI 위험 검토">
    <PhotoPreview photo={finding.photo} label="조치 전 사진" />
    <AnalysisPanel {...analysis} aiMode={finding.aiMode} inspectionTask={finding.inspectionTask}
      result={manual || !successful ? undefined : result} onRun={run} onManual={() => {
      analysis.cancel();
      setManual(true);
      setDescription(finding.description);
      setAction(finding.action);
    }} />
    {(manual || successful) && <Card>
      <Field label="위험 설명" value={description} onChangeText={setDescription} multiline />
      <Field label="권장 조치" value={action} onChangeText={setAction} multiline />
      <Button disabled={analysis.busy} title={manual ? '수동 내용 저장' : '제안 채택 / 수정 내용 저장'} onPress={() => save(false)} />
      {!manual && <Button disabled={analysis.busy} title="AI 제안 거절 · 기존 입력 유지" secondary onPress={() => save(true)} />}
    </Card>}
    {!!error && <Copy tone="error">{error}</Copy>}
  </Page>;
}
