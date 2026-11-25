import { Button, Card, Copy } from '@/components/safelog-ui';
import type { Analysis } from '@/contracts/types';
import type { MockResponse } from './mock';

const modes: MockResponse[] = ['정상', '분석 실패', '읽을 수 없는 결과'];

export function AnalysisPanel({ result, busy, error, onRun, onManual, runDisabled, live }:  {
  runDisabled?: boolean;
  live?: boolean;
  result?: Analysis; busy: boolean; error: string;
  onRun: (mode: MockResponse) => void; onManual: () => void;
}) {
  return <Card>
    <Copy tone="ai">{live ? 'AI 사진 판정' : 'AI 참고 제안 · Mock'}</Copy>
    {busy && <Copy>분석 중…</Copy>}
    {!!error && <Copy tone="error">{error}</Copy>}
    {result && <>
      <Copy tone="ai">{result.assessment}</Copy>
      <Copy>{result.action}</Copy>
      {result.category && <Copy>분야 · {result.category}</Copy>}
      {result.risk !== undefined && <Copy>위험 등급 {result.risk}/5</Copy>}
      <Copy>{result.model} · {result.prompt} · 신뢰도 {result.confidence}</Copy>
    </>}
    {!busy && (live ? <Button title="사진 다시 분석" secondary onPress={() => onRun('정상')} /> :
      modes.map(mode => <Button key={mode} title={mode === '정상' ? 'Mock 분석 실행' : mode + ' 시연'}
        secondary disabled={runDisabled && mode === '정상'} onPress={() => onRun(mode)} />))}
    <Button title={busy ? '분석 취소 · 직접 검토' : 'AI 없이 직접 검토'} secondary onPress={onManual} />
  </Card>;
}
