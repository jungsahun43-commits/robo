import { File } from 'expo-file-system';
import type { Photo, Suggestion } from '@/contracts/types';
import { inspectionCategoryIds, type InspectionCategory } from './inspection-categories';

// The endpoint accepts an original photo and selects a category without a task hint.
const endpoint = process.env.EXPO_PUBLIC_SAFELOG_CLASSIFIER_URL?.trim();
export const photoClassifierConfigured = Boolean(endpoint);

export async function classifyPhoto(photo: Photo, signal: AbortSignal): Promise<Suggestion> {
  if (!endpoint) throw new Error('사진 분류 서버가 설정되지 않았습니다.');
  const imageBase64 = photo.uri.startsWith('data:') && photo.uri.includes(';base64,')
    ? photo.uri.slice(photo.uri.indexOf(',') + 1)
    : await new File(photo.uri).base64();
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageBase64, mimeType: photo.mimeType }),
    signal,
  });
  if (!response.ok) throw new Error('사진 분석 요청에 실패했습니다. (HTTP ' + response.status + ')');
  const data = await response.json();
  if (!data || !inspectionCategoryIds.includes(data.category as InspectionCategory)) {
    throw new Error('사진에서 지원하는 점검 대상을 판정하지 못했습니다.');
  }
  const { category, model, confidence, assessment, action } = data;
  if (typeof model !== 'string' || !model.trim() ||
    typeof confidence !== 'number' || !Number.isFinite(confidence) || confidence < 0 || confidence > 1 ||
    typeof assessment !== 'string' || !assessment.trim() || typeof action !== 'string' || !action.trim()) {
    throw new Error('사진 분류 서버 응답 형식이 올바르지 않습니다.');
  }
  return { category, model, confidence, assessment, action, prompt: 'photo-classifier-v1' };
}
