import { useEffect, useEffectEvent, useState } from 'react';
import { router } from 'expo-router';
import { Button, Card, Copy, Field, message, Page, PhotoPreview } from '@/components/safelog-ui';
import { AnalysisPanel } from '@/features/ai/analysis-panel';
import type { MockResponse } from '@/features/ai/mock';
import { inspectionCategories, inspectionCategoryIds, type InspectionCategory } from '@/features/ai/inspection-categories';
import { useAnalysis } from '@/features/ai/use-analysis';
import { classifyPhoto, photoClassifierConfigured } from '@/features/ai/photo-classifier';
import { workflow } from '@/features/auth/session';
import { useFinding } from '@/features/workflow/use-finding';

export default function AiHazardReviewPage() {
  const { finding, user } = useFinding();
  const [category, setCategory] = useState<InspectionCategory>();
  const analysis = useAnalysis();
  const latest = finding?.analyses.filter(item => item.type === 'BeforeHazard').at(-1);
  const pending = latest?.decision === 'Pending' ? latest : undefined;
  const [manual, setManual] = useState(false);
  const [description, setDescription] = useState(pending?.assessment ?? finding?.description ?? '');
  const [action, setAction] = useState(pending?.action ?? finding?.action ?? '');
  const [error, setError] = useState('');

  function run(mode: MockResponse) {
    if (!finding || user?.role !== 'Inspector' || finding.inspectorId !== user.id || finding.status !== 'Open' ||
      (!photoClassifierConfigured && mode === '정상' && !category)) return;
    setManual(false);
    void analysis.run(async (analyzer, signal) => {
      const suggestion = await (photoClassifierConfigured ? classifyPhoto(finding.photo, signal) : analyzer.hazard(finding.photo, signal));
      if (signal.aborted) throw new Error('분석 취소');
      const result = workflow.execute(user, finding.id, { kind: 'suggest', type: 'BeforeHazard', suggestion });
      if (!('type' in result)) throw new Error('분석 기록을 저장할 수 없습니다.');
      setDescription(result.assessment);
      setAction(result.action);
      return result;
    }, mode, category);
  }

  const analyzeOnOpen = useEffectEvent(() => {
    if (photoClassifierConfigured && finding?.status === 'Open' && !latest) run('정상');
  });
  const findingId = finding?.id;
  const userId = user?.id;
  useEffect(() => {
    const timer = setTimeout(() => analyzeOnOpen(), 0);
    return () => clearTimeout(timer);
  }, [findingId, userId]);

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
    {!photoClassifierConfigured && <Card>
      <Copy>사진 자동 분류 서버 미설정 · Mock 분석 분야</Copy>
      <Copy>시연할 사진 분야를 선택한 다음 Mock 분석을 실행하세요.</Copy>
      {inspectionCategoryIds.map(id => <Button key={id} title={(category === id ? '선택됨 · ' : '') + inspectionCategories[id].label}
        secondary={category !== id} disabled={analysis.busy} onPress={() => setCategory(id)} />)}
    </Card>}
    <AnalysisPanel {...analysis} live={photoClassifierConfigured} runDisabled={!category} result={manual || !successful ? undefined : result} onRun={run} onManual={() => {
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
