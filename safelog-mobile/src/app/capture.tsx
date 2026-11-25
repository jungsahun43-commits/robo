import { useRef, useState } from 'react';
import { router } from 'expo-router';
import { Button, Card, Copy, Field, message, Page } from '@/components/safelog-ui';
import type { Photo } from '@/contracts/types';
import { useSession, workflow } from '@/features/auth/session';
import { site } from '@/features/auth/demo';
import { PhotoField } from '@/features/capture/photo-field';

export default function FindingCapturePage() {
  const { user } = useSession();
  const submitted = useRef(false);
  const [photo, setPhoto] = useState<Photo>();
  const [location, setLocation] = useState('');
  const [description, setDescription] = useState('');
  const [action, setAction] = useState('');
  const [error, setError] = useState('');

  function register() {
    if (submitted.current) return;
    submitted.current = true;
    try {
      if (!user || !photo) throw new Error('점검 사진을 먼저 선택하세요.');
      const finding = workflow.create(user, { photo, location, description, action });
      router.replace({ pathname: '/hazard', params: { id: finding.id } });
    } catch (error) {
      submitted.current = false;
      setError(message(error));
    }
  }

  return <Page title="안전점검 등록">
    <Card>
      <Copy>점검 대상 · {site.name}</Copy>
      <Button title="현장 정보 보기" secondary onPress={() => router.push("/facilities")} />
    </Card>
    <Card><PhotoField label="조치 전 사진" photo={photo} onChange={setPhoto} /></Card>
    <Card>
      <Field label="점검 위치" value={location} onChangeText={setLocation} />
      <Field label="발견 메모" value={description} onChangeText={setDescription} multiline />
      <Field label="수동 권장 조치" value={action} onChangeText={setAction} multiline />
    </Card>
    {!!error && <Copy tone="error">{error}</Copy>}
    <Copy>등록하면 미조치 상태로 저장되고 AI 검토가 열립니다.</Copy>
    <Button title="등록 후 AI 검토" onPress={register} />
  </Page>;
}
