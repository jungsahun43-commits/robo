import { useRef, useState } from 'react';
import { Platform } from 'react-native';
import { Button, Card, Copy, message, Page, PhotoPreview } from '@/components/safelog-ui';
import { exportReport, reportFor } from '@/features/reporting/report';
import { useFinding } from '@/features/workflow/use-finding';

export default function ReportPreviewPage() {
  const { user, finding } = useFinding();
  const active = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  if (!finding || !user || user.role === 'Assignee') return <Page title="안전점검 보고서"><Copy>볼 수 있는 보고서가 없습니다.</Copy></Page>;
  const report = reportFor(finding);

  async function share() {
    if (!finding || active.current) return;
    active.current = true;
    setBusy(true);
    setError('');
    try {
      await exportReport(finding);
    } catch (error) {
      setError(message(error));
    } finally {
      active.current = false;
      setBusy(false);
    }
  }

  return <Page title="SafeLog AI 안전점검 보고서">
    <Card><Copy>{report.header.join('\n')}</Copy></Card>
    <Card>
      <Copy>점검 기록</Copy>
      <Copy>{report.record.join('\n')}</Copy>
      {report.photos.map(({ label, photo }) => <PhotoPreview key={label} label={label} photo={photo} />)}
    </Card>
    <Card>
      <Copy tone="ai">AI 참고 제안 및 검토 결정</Copy>
      {report.ai.length ? report.ai.map(item => <Copy key={item.id} tone="ai">{item.text}</Copy>) : <Copy>AI 제안 없음</Copy>}
    </Card>
    <Card>
      <Copy tone="human">사람의 조치 및 최종 판단</Copy>
      {report.logs.map((text, index) => <Copy key={index}>{text}</Copy>)}
    </Card>
    {!!error && <Copy tone="error">{error}</Copy>}
    <Button title={busy ? '보고서 생성 중…' : Platform.OS === 'web' ? 'HTML 보고서 다운로드' : 'PDF 생성 및 공유'}
      disabled={busy} onPress={() => { void share(); }} />
  </Page>;
}
