import { File, Paths } from 'expo-file-system';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';
import type { Finding, Photo } from '@/contracts/types';
import { profiles, site } from '@/features/auth/demo';
import { assessmentLines, missingAssessmentFields, type Assessment } from './assessment';
import { escapeHtml, preliminarySheetCss, preliminarySheetHtml, latestInspectionCategory, type Checklist } from './checklist';

export type SheetDetails = { contractorParticipation?: string; workDescription?: string };

const name = (id?: string) => profiles.find(profile => profile.id === id)?.name ?? id ?? '미배정';

export function reportFor(finding: Finding, assessment: Assessment, details: SheetDetails = {}) {
  const category = latestInspectionCategory(finding.analyses);
  const part = category ? category.label + (category.mock ? ' (Mock 시연)' : ' (AI 판정)') : '미판정';
  return {
    title: finding.status === 'Verified' ? '완료 안전점검 보고서' : '예비 안전점검 보고서',
    assessment: assessmentLines(assessment),
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
    header: ['시연용 사업장 · ' + site.name, site.address, '점검자 · ' + name(finding.inspectorId), '점검 일시 · ' + finding.createdAt],
    record: [
      '기록 ID · ' + finding.id,
      '점검 ID · ' + finding.inspectionId,
      '장소 · ' + finding.location,
      '발견 내용 · ' + finding.description,
      '권장 조치 · ' + finding.action,
      '담당자 · ' + name(finding.assigneeId),
      '위험 등급 · ' + (finding.risk ?? '미지정'),
      '작업 상태 · ' + finding.status,
    ],
    photos: [{ label: '조치 전 사진', photo: finding.photo }, ...finding.after.map((photo, index) => ({
      label: '조치 후 사진 ' + (index + 1), photo,
    }))],
    ai: finding.analyses.map(item => ({
      id: item.id,
      text: [item.type + ' · ' + item.model + ' / ' + item.prompt, item.assessment, item.action,
        '신뢰도 · ' + item.confidence, '검토 · ' + item.decision + (item.reviewer ? ' · ' + name(item.reviewer) : ''), item.at].join('\n'),
    })),
    logs: finding.logs.map(item => [item.action + ' · ' + name(item.actor), item.note, item.at].filter(Boolean).join('\n')),
  };
}

const paragraph = (text: string) => '<p>' + escapeHtml(text) + '</p>';

async function image(photo: Photo) {
  if (!/^image\/[a-z0-9.+-]+$/i.test(photo.mimeType)) throw new Error('PDF에 넣을 수 없는 이미지 형식입니다.');
  if (Platform.OS === 'web' && /^data:image\//.test(photo.uri)) return photo.uri;
  return 'data:' + photo.mimeType + ';base64,' + await new File(photo.uri).base64();
}

async function completedReportHtml(report: ReturnType<typeof reportFor>) {
  const images = await Promise.all(report.photos.map(async ({ label, photo }) =>
    '<figure><figcaption>' + escapeHtml(label) + '</figcaption><img src="' + escapeHtml(await image(photo)) + '" /></figure>'));
  return '<h1>' + escapeHtml(report.title) + '</h1>' + report.header.map(paragraph).join('') +
    '<h2>점검 기록</h2>' + report.record.map(paragraph).join('') + images.join('') +
    '<section class="ai"><h2>AI 참고 제안 및 검토 결정</h2>' + (report.ai.map(item => paragraph(item.text)).join('') || paragraph('AI 제안 없음')) +
    '</section><section class="human"><h2>사람의 조치 및 최종 판단</h2>' + report.logs.map(paragraph).join('') +
    '</section>' +
    '<h2>위험성평가 기록</h2>' + report.assessment.map(paragraph).join('');
}

export async function exportReport(finding: Finding, assessment: Assessment, checklist: Checklist, details: SheetDetails) {
  const missing = missingAssessmentFields(assessment);
  if (finding.status === 'Verified' && missing.length) {
    throw new Error('완료 보고서의 필수 입력 항목: ' + missing.map(field => field.label).join(', '));
  }
  const report = reportFor(finding, assessment, details);
  const preliminary = finding.status !== 'Verified';
  const filename = 'safelog-' + (finding.status === 'Verified' ? 'final-' : 'preliminary-') + finding.id;
  const html = '<!doctype html><html lang="ko"><meta charset="utf-8"><style>' +
    'body{font:15px sans-serif;color:#243848;margin:28px}p{white-space:pre-wrap}img{max-width:100%;max-height:340px}' +
    'figure{break-inside:avoid;margin:16px 0}.ai{color:#1e58a5}.human{color:#126b58}' +
    preliminarySheetCss + '</style><body>' +
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
