import { Button, Card, Copy } from '@/components/safelog-ui';
import type { AiMode, Analysis, InspectionTask } from '@/contracts/types';
import type { MockResponse } from './mock';
import { taskLabel, taskOption, photoRatesLabel } from './tasks';
import { isPcbTask } from '@/contracts/inspection';

const modes: MockResponse[] = ['정상', '분석 실패', '읽을 수 없는 결과'];

export function AnalysisPanel({ result, busy, error, onRun, onManual, aiMode = 'mock', inspectionTask, comparison = false }: {
  result?: Analysis; busy: boolean; error: string;
  onRun: (mode: MockResponse) => void; onManual: () => void;
  aiMode?: AiMode; inspectionTask?: InspectionTask; comparison?: boolean;
}) {
  const real = aiMode === 'fixed-cpu';
  const direct = aiMode === 'manual';
  const unsupportedComparison = real && comparison && isPcbTask(inspectionTask);
  return <Card>
    <Copy tone={direct ? 'human' : 'ai'}>{direct ? 'AI 없이 직접 검토 · ' + taskLabel(inspectionTask)
      : 'AI 참고 제안 · ' + (real ? '고정 CPU 모델 · ' + taskLabel(inspectionTask) : '모의 시연(Mock)')}</Copy>
    {(real || direct) && <Copy>{taskOption(inspectionTask)?.scope}</Copy>}
    {real && <Copy>{photoRatesLabel(inspectionTask)} · 현장 성능·위치 정확도 보증이 아닙니다.</Copy>}
    {unsupportedComparison && <Copy>PCB는 사진별 외관 후보 확인을 지원합니다. 조치 전후는 사진을 직접 비교해 검토하세요.</Copy>}
    {busy && <Copy>분석 중…</Copy>}
    {!!error && <Copy tone="error">{error}</Copy>}
    {result && !busy && !error && <>
      {result.candidateDetected !== undefined && <Copy tone="ai">{result.candidateDetected
        ? '손상 후보 감지 · 사진을 직접 확인하세요.' : '후보 미검출 · 실제 안전이나 무결함을 보증하지 않습니다.'}</Copy>}
      <Copy tone="ai">{result.assessment}</Copy>
      <Copy>{result.action}</Copy>
      {result.risk !== undefined && <Copy>{result.category} · 위험 등급 {result.risk}/5</Copy>}
      <Copy>{result.model} · {result.prompt}</Copy>
      {result.anomalyScore !== undefined && <Copy>AI 참고 점수 {result.anomalyScore.toFixed(4)} · 기준 {result.threshold?.toFixed(4)} (확률 아님)</Copy>}
      {result.anomalyScore === undefined && result.confidence !== undefined && !(real && comparison) &&
        <Copy>AI 참고 점수 {result.confidence.toFixed(4)} (확률 아님)</Copy>}
      {result.modelSha256 && <Copy>모델 식별 · {result.modelSha256.slice(0, 12)}</Copy>}
      {isPcbTask(inspectionTask) && <Copy>정밀 결함 위치나 내부 전기 고장 판정은 제공하지 않습니다.</Copy>}
    </>}
    {!busy && real && !unsupportedComparison && <Button title={comparison ? '실제 모델로 전후 비교' : '실제 모델로 사진 분석'} onPress={() => onRun('정상')} />}
    {!busy && !real && !direct && modes.map(mode => <Button key={mode} title={mode === '정상' ? 'Mock 분석 실행' : mode + ' 시연'}
      secondary onPress={() => onRun(mode)} />)}
    <Button title={busy ? '분석 취소 · 직접 검토' : 'AI 없이 직접 검토'} secondary onPress={onManual} />
  </Card>;
}
