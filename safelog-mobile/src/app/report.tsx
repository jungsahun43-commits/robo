import { useRef, useState } from 'react';
import { Platform } from 'react-native';
import { Button, Card, Copy, Field, message, Page, PhotoPreview } from '@/components/safelog-ui';
import { exportReport, reportFor, type SheetDetails } from '@/features/reporting/report';
import { useFinding } from '@/features/workflow/use-finding';
import type { Finding } from '@/contracts/types';
import { assessmentFields, type Assessment } from '@/features/reporting/assessment';
import { checklistWithAi, type Checklist } from '@/features/reporting/checklist';
import { PreliminarySheet } from '@/features/reporting/preliminary-sheet';

export default function ReportPreviewPage() {
  const { user, finding } = useFinding();
  if (!finding || !user || user.role === 'Assignee') return <Page title="안전점검 보고서"><Copy>볼 수 있는 보고서가 없습니다.</Copy></Page>;
  return <ReportForm key={finding.id} finding={finding} />;
}

function ReportForm({ finding }: { finding: Finding }) {
  const [assessment, setAssessment] = useState<Assessment>({});
  const [manualChecklist, setChecklist] = useState<Checklist>({});
  const checklist = checklistWithAi(finding.analyses, manualChecklist);
  const [details, setDetails] = useState<SheetDetails>({});
  const active = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const report = reportFor(finding, assessment, details);
  const preliminary = finding.status !== 'Verified';

  async function share() {
    if (!finding || active.current) return;
    active.current = true;
    setBusy(true);
    setError('');
    try {
      await exportReport(finding, assessment, checklist, details);
    } catch (error) {
      setError(message(error));
    } finally {
      active.current = false;
      setBusy(false);
    }
  }

  return <Page title={report.title}>
    {preliminary ? <>
      <PreliminarySheet info={report.sheetInfo} checklist={checklist} onChange={(id, update) =>
        setChecklist(current => ({ ...current, [id]: { ...current[id], ...update } }))} />
      <Card>
        <Copy>점검표 현장 정보</Copy>
        <Field label="관계수급인 참여 여부 및 비고" value={details.contractorParticipation ?? ''}
          onChangeText={contractorParticipation => setDetails(current => ({ ...current, contractorParticipation }))} />
        <Field label="작업내용" value={details.workDescription ?? ''} multiline
          onChangeText={workDescription => setDetails(current => ({ ...current, workDescription }))} />
      </Card>
    </> : <>
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
      <Card>
        <Copy>위험성평가 기록</Copy>
        <Copy>완료 보고서를 출력하려면 개선대책 이행 결과를 입력하세요. 입력 내용은 현재 화면에서 유지됩니다.</Copy>
        {assessmentFields.map(({ key, label }) => <Field key={key} label={label} value={assessment[key] ?? ''} multiline
          onChangeText={value => setAssessment(current => ({ ...current, [key]: value }))} />)}
      </Card>
    </>}
    {!!error && <Copy tone="error">{error}</Copy>}
    <Button title={busy ? '보고서 생성 중…' : Platform.OS === 'web' ? 'HTML 보고서 다운로드' : 'PDF 생성 및 공유'}
      disabled={busy} onPress={() => { void share(); }} />
  </Page>;
}
