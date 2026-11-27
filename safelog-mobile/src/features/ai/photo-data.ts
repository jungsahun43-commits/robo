import { File } from 'expo-file-system';
import { Platform } from 'react-native';
import type { Photo } from '@/contracts/types';

const maxBytes = 8 * 1024 * 1024;
const maxBase64Length = Math.ceil(maxBytes / 3) * 4;

function checkActive(signal: AbortSignal) {
  if (signal.aborted) throw new Error('사진 읽기를 취소했습니다.');
}

function dataUrl(base64: string) {
  if (!base64 || base64.length > maxBase64Length || base64.length % 4 !== 0 ||
      !/^[A-Za-z0-9+/]*={0,2}$/.test(base64)) {
    throw new Error('8MiB 이하 JPEG 또는 PNG 사진을 선택하세요.');
  }
  const padding = base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0;
  if (base64.length / 4 * 3 - padding > maxBytes) throw new Error('사진은 8MiB 이하여야 합니다.');
  // Inspect actual encoded bytes instead of trusting an unknown picker MIME.
  const mime = base64.startsWith('/9j/') ? 'image/jpeg'
    : base64.startsWith('iVBORw0KGgo') ? 'image/png' : undefined;
  if (!mime) throw new Error('실제 JPEG 또는 PNG 파일을 선택하세요. HEIC는 지원하지 않습니다.');
  return 'data:' + mime + ';base64,' + base64;
}

async function webBase64(uri: string, signal: AbortSignal) {
  if (!uri.startsWith('blob:')) throw new Error('이 앱에서 선택한 로컬 사진만 분석할 수 있습니다.');
  const response = await fetch(uri, { signal });
  if (!response.ok) throw new Error('선택한 사진을 읽을 수 없습니다.');
  const blob = await response.blob();
  if (!blob.size || blob.size > maxBytes) throw new Error('사진은 8MiB 이하여야 합니다.');
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    const finish = () => signal.removeEventListener('abort', cancel);
    const cancel = () => { reader.abort(); finish(); reject(new Error('사진 읽기를 취소했습니다.')); };
    reader.onload = () => { finish(); resolve(String(reader.result).split(',')[1] ?? ''); };
    reader.onerror = () => { finish(); reject(new Error('선택한 사진을 읽을 수 없습니다.')); };
    reader.onabort = () => { finish(); reject(new Error('사진 읽기를 취소했습니다.')); };
    if (signal.aborted) { cancel(); return; }
    signal.addEventListener('abort', cancel, { once: true });
    reader.readAsDataURL(blob);
  });
}

export async function readPhotoDataUrl(photo: Photo, signal: AbortSignal, maxPixels = 40_000_000) {
  checkActive(signal);
  if (photo.fileSize !== undefined && photo.fileSize > maxBytes) throw new Error('사진은 8MiB 이하여야 합니다.');
  if (photo.width && photo.height && photo.width * photo.height > maxPixels) {
    throw new Error(`사진 크기는 ${maxPixels / 1_000_000}MP 이하여야 합니다. 더 작은 JPEG/PNG를 선택하세요.`);
  }
  let base64: string;
  if (photo.uri.startsWith('data:')) {
    const match = /^data:image\/(?:jpeg|png);base64,([A-Za-z0-9+/=]+)$/.exec(photo.uri);
    if (!match) throw new Error('JPEG 또는 PNG 사진을 선택하세요.');
    base64 = match[1];
  } else if (Platform.OS === 'web') {
    base64 = await webBase64(photo.uri, signal);
  } else {
    if (!/^(?:file|content):\/\//.test(photo.uri)) throw new Error('로컬 사진 파일을 선택하세요.');
    const file = new File(photo.uri);
    if (!file.size || file.size > maxBytes) throw new Error('읽을 수 있는 8MiB 이하 사진을 선택하세요.');
    base64 = await file.base64();
  }
  checkActive(signal);
  return dataUrl(base64);
}
