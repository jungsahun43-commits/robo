import { File, Paths } from 'expo-file-system';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';
import type { Finding, Profile } from '@/contracts/types';
import { readPhotoDataUrl } from '@/features/ai/photo-data';

import { taskLabel } from '@/features/ai/tasks';
import { profiles, site } from '@/features/auth/demo';
import { missingAssessmentFields, type Assessment } from './assessment';
import { escapeHtml, preliminarySheetCss, preliminarySheetHtml, latestInspectionCategory, type Checklist } from './checklist';

export const implementationOptions = ['완료', '부분 이행', '미이행', '해당 없음'] as const;
export const judgmentOptions = ['적정', '부적정', '재점검 필요'] as const;
export const reviewOptions = ['수용', '수정 수용', '미수용', '추가 확인'] as const;
export const completedReportFooter = '※ 점검 건별 1부 작성. 미완료 또는 잔여 위험은 후속 조치 대상으로 관리합니다.';

export type SheetDetails = {
  contractorParticipation?: string;
  workDescription?: string;
  department?: string;
  reviewNote?: string;
  beforeTakenAt?: string;
  afterTakenAt?: string;
  assessmentNumber?: string;
  implementationStatus?: typeof implementationOptions[number];
  finalJudgment?: typeof judgmentOptions[number];
  signature?: string;
};

type ReportField = { label: string; value?: string; options?: readonly string[] };
type ReportRow = readonly ReportField[];

const field = (label: string, value?: string, options?: readonly string[]): ReportField => ({ label, value, options });
const date = (value?: string) => value ? new Date(value).toLocaleDateString('ko-KR') : undefined;

export function reportFieldText({ value, options }: ReportField) {
  return options ? options.map(option => (option === value ? '☑ ' : '□ ') + option).join('　') : value?.trim() || '—';
}

export function reportFor(finding: Finding, assessment: Assessment, details: SheetDetails = {}, people: readonly Profile[] = profiles) {
  const name = (id?: string) => people.find(profile => profile.id === id)?.name ?? id ?? '미배정';
  const category = latestInspectionCategory(finding.analyses);
  const part = category ? category.label + (category.mock ? ' (Mock 시연)' : ' (AI 판정)') : '미판정';
  return {
    title: finding.status === 'Verified' ? '완료 안전점검 보고서' : '예비 안전점검 보고서',
    sheetInfo: [
      ['점검일자', new Date(finding.createdAt).toLocaleDateString('ko-KR')],
      ['점검 대상 (Part)', part],
      ['현장명', site.name],
      ['회사명', '세이프 금속 (시연)'],
      ['주소', site.address],
      ['점검자', name(finding.inspectorId)],
      ['관계수급인 참여', details.contractorParticipation?.trim() || '미기록'],
      ['참여 근로자', assessment.workers?.trim() || '미입력'],
      ['근로자대표', assessment.representative?.trim() || '미입력'],
      ['작업내용', details.workDescription?.trim() || '미기록'],
    ] as const,
    completed: completedReportFor(finding, assessment, details, name, category?.label),
  };
}

function completedReportFor(finding: Finding, assessment: Assessment, details: SheetDetails,
  name: (id?: string) => string, category?: string) {
  const verified = finding.logs.filter(log => log.action === 'verified').at(-1);
  const submissions = finding.logs.filter(log => log.action === 'action_submitted');
  const hazard = finding.analyses.filter(item => item.type === 'BeforeHazard').at(-1);
  const comparison = finding.analyses.filter(item => item.type === 'AfterComparison').at(-1);
  const review = hazard ? { Accepted: '수용', Edited: '수정 수용', Rejected: '미수용', Pending: undefined }[hazard.decision] : undefined;
  const metadata: ReportRow[] = [
    [field('보고서 번호', '완료-' + finding.id), field('예비보고서 번호', '예비-' + finding.id)],
    [field('회사명', '세이프 금속 (시연)'), field('현장명', site.name)],
    [field('점검일자', date(finding.createdAt)), field('완료보고일', date(verified?.at))],
    [field('점검자', name(finding.inspectorId)), field('담당자 / 부서', [name(finding.assigneeId), details.department?.trim()].filter(Boolean).join(' / '))],
  ];
  const sections: { title: string; note?: string; rows: ReportRow[] }[] = [
    { title: '01 점검 및 발견 내용', rows: [
      [field('점검 위치', finding.location), field('검사 분야', category ?? (finding.inspectionTask ? taskLabel(finding.inspectionTask) : '미판정'))],
      [field('발견 내용', finding.description)],
      [field('권장 조치', finding.action)],
      [field('작업 상태', '조치 완료', ['미착수', '진행 중', '조치 완료', '보류'])],
    ] },
    { title: '02 AI 참고 제안 및 사람의 검토', note: 'AI 결과는 참고용이며 최종 판정은 사람이 수행합니다.', rows: [
      [field('AI 참고 제안', [hazard, comparison].flatMap(item => item ? [item.assessment, item.action] : []).join('\n') || 'AI 제안 없음')],
      [field('사람의 검토 결정', review, reviewOptions)],
      [field('검토 근거 / 의견', [details.reviewNote, verified?.note].filter(Boolean).join('\n'))],
    ] },
    { title: '03 조치 전후 사진 및 실제 조치', rows: [
      [field('사람의 실제 조치', submissions.map(log => log.note).filter(Boolean).join('\n'))],
    ] },
    { title: '04 위험성평가 개선대책 이행 결과 및 최종 판단', rows: [
      [field('위험성평가 번호', details.assessmentNumber), field('조치 완료일', date(submissions.at(-1)?.at))],
      [field('개선대책 이행', details.implementationStatus, implementationOptions)],
      [field('효과 / 잔여 위험', assessment.implementation)],
      [field('사람의 최종 판단', details.finalJudgment, judgmentOptions)],
      [field('최종 확인자', verified ? name(verified.actor) : undefined),
        field('확인일 / 서명', [date(verified?.at), details.signature?.trim()].filter(Boolean).join(' / '))],
    ] },
  ];
  return { metadata, sections, photos: [
    { label: '조치 전 사진', photo: finding.photo, takenAt: details.beforeTakenAt },
    ...finding.after.map((photo, index) => ({
      label: finding.after.length > 1 ? '조치 후 사진 ' + (index + 1) : '조치 후 사진',
      photo, takenAt: index === finding.after.length - 1 ? details.afterTakenAt : undefined,
    })),
  ] };
}

const tableHtml = (rows: ReportRow[]) => '<table><tbody>' + rows.map(row => '<tr>' +
  row.map(item => '<th scope="row">' + escapeHtml(item.label) + '</th><td' +
    (row.length === 1 ? ' colspan="3"' : '') + '>' + escapeHtml(reportFieldText(item)) + '</td>').join('') +
  '</tr>').join('') + '</tbody></table>';

async function completedReportHtml(report: ReturnType<typeof reportFor>) {
  const { metadata, sections, photos } = report.completed;
  const images = await Promise.all(photos.map(async ({ label, photo, takenAt }) =>
    '<figure><figcaption>' + escapeHtml(label) + '</figcaption><img src="' +
    escapeHtml(await readPhotoDataUrl(photo, new AbortController().signal)) + '" alt="' + escapeHtml(label) +
    '" /><div>촬영일시: ' + escapeHtml(takenAt?.trim() || '—') + '</div></figure>'));
  return '<article class="completed"><header><div>SAFELOG / 설비·시설 안전점검 <span>문서 유형 | 완료보고서</span></div>' +
    '<h1>' + escapeHtml(report.title) + '</h1><p>점검 결과 · 개선조치 · 이행 확인 기록</p></header>' + tableHtml(metadata) +
    sections.map(({ title, note, rows }, index) => '<section><h2>' + escapeHtml(title) + '</h2>' +
      (note ? '<p class="form-note">' + escapeHtml(note) + '</p>' : '') +
      (index === 2 ? '<div class="photos">' + images.join('') + '</div>' : '') + tableHtml(rows) + '</section>').join('') +
    '<footer>' + escapeHtml(completedReportFooter) + '</footer></article>';
}

const completedReportCss = '.completed{font-size:11px;color:#192943}.completed header div{font-weight:bold;border-bottom:2px solid #254269;padding-bottom:5px}' +
  '.completed header span{float:right}.completed h1{font-size:24px;margin:8px 0 2px}.completed header p{margin:0 0 10px}' +
  '.completed h2{font-size:13px;background:#e4ebf4;padding:5px 8px;margin:10px 0 0;break-after:avoid;-webkit-print-color-adjust:exact;print-color-adjust:exact}' +
  '.completed table{width:100%;table-layout:fixed;border-collapse:collapse}.completed th,.completed td{border:1px solid #9daec2;padding:5px 7px;white-space:pre-wrap;overflow-wrap:anywhere}' +
  '.completed th{width:17%;text-align:left;background:#f3f6fa;-webkit-print-color-adjust:exact;print-color-adjust:exact}.completed tr{break-inside:avoid}' +
  '.completed .photos{display:flex;flex-wrap:wrap;gap:6px;margin:5px 0}.completed figure{box-sizing:border-box;width:calc(50% - 3px);margin:0;border:1px solid #9daec2;padding:5px;text-align:center}' +
  '.completed img{display:block;width:100%;height:115px;object-fit:contain}.completed .form-note,.completed footer{font-size:10px;margin:5px 0}';

export async function exportReport(finding: Finding, assessment: Assessment, checklist: Checklist, details: SheetDetails, people?: readonly Profile[]) {
  const missing = missingAssessmentFields(assessment);
  if (finding.status === 'Verified' && missing.length) {
    throw new Error('완료 보고서의 필수 입력 항목: ' + missing.map(field => field.label).join(', '));
  }
  const report = reportFor(finding, assessment, details, people);
  const preliminary = finding.status !== 'Verified';
  const filename = 'safelog-' + (finding.status === 'Verified' ? 'final-' : 'preliminary-') + finding.id;
  const html = '<!doctype html><html lang="ko"><meta charset="utf-8"><style>' +
    'body{font:15px sans-serif;color:#243848;margin:28px}p{white-space:pre-wrap}img{max-width:100%;max-height:340px}' +
    'figure{break-inside:avoid;margin:16px 0}.ai{color:#1e58a5}.human{color:#126b58}' +
    preliminarySheetCss + completedReportCss + '</style><body>' +
    (preliminary ? preliminarySheetHtml(report.sheetInfo, checklist) : await completedReportHtml(report)) +
    '</body></html>';

  if (Platform.OS === 'web') {
    const url = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = filename + '.html';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return;
  }
  if (!await Sharing.isAvailableAsync()) throw new Error('이 기기에서 파일 공유를 사용할 수 없습니다.');
  const android = Platform.OS === 'android';
  const generated = await Print.printToFileAsync({ html, base64: android });
  let shareUri = generated.uri;
  if (android) {
    // Expo Go's Print module writes to its host cache. Sharing accepts the experience cache.
    if (!generated.base64) throw new Error('보고서 PDF를 생성할 수 없습니다.');
    const destination = new File(Paths.cache, filename + '.pdf');
    await destination.write(generated.base64, { encoding: 'base64' });
    shareUri = destination.uri;
  }
  await Sharing.shareAsync(shareUri, { mimeType: 'application/pdf', UTI: 'com.adobe.pdf' });
}
