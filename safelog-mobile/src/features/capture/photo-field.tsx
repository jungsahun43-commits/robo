import { useEffect, useRef, useState } from 'react';
import * as Picker from 'expo-image-picker';
import { Platform } from 'react-native';
import type { Photo } from '@/contracts/types';
import { Button, Copy, message, PhotoPreview } from '@/components/safelog-ui';
import { readPhotoDataUrl } from '@/features/ai/photo-data';

export function PhotoField({ label, photo, onChange }: {
  label: string; photo?: Photo; onChange: (photo: Photo) => void;
}) {
  const active = useRef(false);
  const request = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => () => request.current?.abort(), []);

  async function pick(camera: boolean) {
    if (active.current) return;
    active.current = true;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError('');
    try {
      if (camera && Platform.OS !== 'web') {
        const permission = await Picker.requestCameraPermissionsAsync();
        if (!permission.granted) throw new Error('카메라 권한을 허용하거나 갤러리에서 사진을 선택하세요.');
      }
      const options: Picker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.75 };
      const result = await (camera ? Picker.launchCameraAsync(options) : Picker.launchImageLibraryAsync(options));
      if (!controller.signal.aborted && !result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        const next = { uri: asset.uri, mimeType: asset.mimeType ?? 'application/octet-stream',
          width: asset.width, height: asset.height, fileSize: asset.fileSize };
        const encoded = await readPhotoDataUrl(next, controller.signal);
        if (!controller.signal.aborted) onChange({ ...next,
          mimeType: encoded.startsWith('data:image/png;') ? 'image/png' : 'image/jpeg' });
      }
    } catch (error) {
      if (!controller.signal.aborted) setError(message(error));
    } finally {
      active.current = false;
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return <>
    <PhotoPreview photo={photo} label={label} />
    <Copy>JPEG·PNG 사진, 파일당 8MiB 이하를 사용하세요.</Copy>
    <Button title="사진 촬영" secondary disabled={busy} onPress={() => { void pick(true); }} />
    <Button title="갤러리에서 사진 선택" secondary disabled={busy} onPress={() => { void pick(false); }} />
    {!!error && <Copy tone="error">{error}</Copy>}
  </>;
}
