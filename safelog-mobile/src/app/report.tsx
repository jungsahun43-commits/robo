import { useRef, useState } from 'react';
import { Platform } from 'react-native';
import { Button, Card, Copy, Field, message, Page, PhotoPreview } from '@/components/safelog-ui';
import { completedReportFooter, exportReport, implementationOptions, judgmentOptions, reportFieldText, reportFor, type SheetDetails } from '@/features/reporting/report';
import { useFinding } from '@/features/workflow/use-finding';
import { useProfiles } from '@/features/auth/session';
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
  const people = useProfiles();
  const [assessment, setAssessment] = useState<Assessment>({});
  const [manualChecklist, setChecklist] = useState<Checklist>({});
  const checklist = checklistWithAi(finding.analyses, manualChecklist);
  const [details, setDetails] = useState<SheetDetails>({});
  const active = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const report = reportFor(finding, assessment, details, people);
  const preliminary = finding.status !== 'Verified';

  async function share() {
    if (!finding || active.current) return;
    active.current = true;
    setBusy(true);
    setError('');
    try {
      await exportReport(finding, assessment, checklist, details, people);
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
      <Card>
        <Copy>SAFELOG / 설비·시설 안전점검 · 문서 유형 | 완료보고서</Copy>
        <Copy>점검 결과 · 개선조치 · 이행 확인 기록</Copy>
        {report.completed.metadata.flat().map(item => <Copy key={item.label}>{item.label} · {reportFieldText(item)}</Copy>)}
        <Field label="담당 부서" value={details.department ?? ''}
          onChangeText={department => setDetails(current => ({ ...current, department }))} />
      </Card>
      {report.completed.sections.map(({ title, note, rows }, index) => <Card key={title}>
        <Copy>{title}</Copy>
        {note && <Copy tone="ai">{note}</Copy>}
        {index === 2 && report.completed.photos.map(({ label, photo, takenAt }) =>
          <PhotoPreview key={label} label={takenAt ? label + ' · 촬영일시: ' + takenAt : label} photo={photo} />)}
        {rows.flat().map(item => <Copy key={item.label}>{item.label} · {reportFieldText(item)}</Copy>)}
        {index === 1 && <Field label="검토 근거 / 의견" multiline value={details.reviewNote ?? ''}
          onChangeText={reviewNote => setDetails(current => ({ ...current, reviewNote }))} />}
        {index === 2 && <>
          <Field label="조치 전 사진 촬영일시" value={details.beforeTakenAt ?? ''}
            onChangeText={beforeTakenAt => setDetails(current => ({ ...current, beforeTakenAt }))} />
          <Field label="마지막 조치 후 사진 촬영일시" value={details.afterTakenAt ?? ''}
            onChangeText={afterTakenAt => setDetails(current => ({ ...current, afterTakenAt }))} />
        </>}
        {index === 3 && <>
          <Field label="위험성평가 번호" value={details.assessmentNumber ?? ''}
            onChangeText={assessmentNumber => setDetails(current => ({ ...current, assessmentNumber }))} />
          <Copy>개선대책 이행</Copy>
          {implementationOptions.map(implementationStatus => <Button key={implementationStatus} title={implementationStatus}
            secondary={details.implementationStatus !== implementationStatus}
            onPress={() => setDetails(current => ({ ...current, implementationStatus }))} />)}
          {assessmentFields.map(({ key }) => <Field key={key} label="효과 / 잔여 위험" value={assessment[key] ?? ''} multiline
            onChangeText={value => setAssessment(current => ({ ...current, [key]: value }))} />)}
          <Copy>사람의 최종 판단</Copy>
          {judgmentOptions.map(finalJudgment => <Button key={finalJudgment} title={finalJudgment}
            secondary={details.finalJudgment !== finalJudgment}
            onPress={() => setDetails(current => ({ ...current, finalJudgment }))} />)}
          <Field label="서명 이니셜 또는 이름" value={details.signature ?? ''}
            onChangeText={signature => setDetails(current => ({ ...current, signature }))} />
        </>}
      </Card>)}
      <Copy>{completedReportFooter}</Copy>
    </>}
    {!!error && <Copy tone="error">{error}</Copy>}
    <Button title={busy ? '보고서 생성 중…' : Platform.OS === 'web' ? 'HTML 보고서 다운로드' : 'PDF 생성 및 공유'}
      disabled={busy} onPress={() => { void share(); }} />
  </Page>;
}
