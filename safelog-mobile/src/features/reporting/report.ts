import { File, Paths } from 'expo-file-system';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';
import type { Finding, Photo, Profile } from '@/contracts/types';
import { readPhotoDataUrl } from '@/features/ai/photo-data';
import { reportFor } from './report-data';
export { reportFor } from './report-data';


const escape = (text: string) => text.replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[char] ?? char);
const paragraph = (text: string) => '<p>' + escape(text) + '</p>';

async function image(photo: Photo) {
  return readPhotoDataUrl(photo, new AbortController().signal);
}

export async function exportReport(finding: Finding, people?: readonly Profile[]) {
  const report = reportFor(finding, people);
  const images = await Promise.all(report.photos.map(async ({ label, photo }) =>
    '<figure><figcaption>' + escape(label) + '</figcaption><img src="' + escape(await image(photo)) + '" /></figure>'));
  const html = '<!doctype html><html lang="ko"><meta charset="utf-8"><style>' +
    'body{font:15px sans-serif;color:#243848;margin:28px}p{white-space:pre-wrap}img{max-width:100%;max-height:340px}' +
    'figure{break-inside:avoid;margin:16px 0}.ai{color:#1e58a5}.human{color:#126b58}</style><body>' +
    '<h1>SafeLog AI 안전점검 보고서</h1>' + report.header.map(paragraph).join('') +
    '<h2>점검 기록</h2>' + report.record.map(paragraph).join('') + images.join('') +
    '<section class="ai"><h2>AI 참고 제안 및 검토 결정</h2>' + (report.ai.map(item => paragraph(item.text)).join('') || paragraph('AI 제안 없음')) +
    '</section><section class="human"><h2>사람의 조치 및 최종 판단</h2>' + report.logs.map(paragraph).join('') +
    '</section></body></html>';
  if (Platform.OS === 'web') {
    const url = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'safelog-' + finding.id + '.html';
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
    const destination = new File(Paths.cache, 'safelog-' + finding.id + '.pdf');
    await destination.write(generated.base64, { encoding: 'base64' });
    shareUri = destination.uri;
  }
  await Sharing.shareAsync(shareUri, { mimeType: 'application/pdf', UTI: 'com.adobe.pdf' });
}
