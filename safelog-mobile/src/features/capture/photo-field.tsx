import { useRef, useState } from 'react';
import * as Picker from 'expo-image-picker';
import { Platform } from 'react-native';
import type { Photo } from '@/contracts/types';
import { Button, Copy, message, PhotoPreview } from '@/components/safelog-ui';

export function PhotoField({ label, photo, onChange }: {
  label: string; photo?: Photo; onChange: (photo: Photo) => void;
}) {
  const active = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function pick(camera: boolean) {
    if (active.current) return;
    active.current = true;
    setBusy(true);
    setError('');
    try {
      if (camera && Platform.OS !== 'web') {
        const permission = await Picker.requestCameraPermissionsAsync();
        if (!permission.granted) throw new Error('카메라 권한을 허용하거나 갤러리에서 사진을 선택하세요.');
      }
      const options: Picker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.75 };
      const result = await (camera ? Picker.launchCameraAsync(options) : Picker.launchImageLibraryAsync(options));
      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        onChange({ uri: asset.uri, mimeType: asset.mimeType ?? 'image/jpeg' });
      }
    } catch (error) {
      setError(message(error));
    } finally {
      active.current = false;
      setBusy(false);
    }
  }

  return <>
    <PhotoPreview photo={photo} label={label} />
    <Button title="사진 촬영" secondary disabled={busy} onPress={() => { void pick(true); }} />
    <Button title="갤러리에서 사진 선택" secondary disabled={busy} onPress={() => { void pick(false); }} />
    {!!error && <Copy tone="error">{error}</Copy>}
  </>;
}
