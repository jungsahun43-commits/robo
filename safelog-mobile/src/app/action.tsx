import { useState } from 'react';
import { router } from 'expo-router';
import { Button, Card, Copy, Field, message, Page } from '@/components/safelog-ui';
import type { Photo } from '@/contracts/types';
import { workflow } from '@/features/auth/session';
import { PhotoField } from '@/features/capture/photo-field';
import { useFinding } from '@/features/workflow/use-finding';

export default function ActionSubmitPage() {
  const { user, finding } = useFinding();
  const [note, setNote] = useState('');
  const [memo, setMemo] = useState('');
  const [photo, setPhoto] = useState<Photo>();
  const [error, setError] = useState('');
  if (!finding || !user || finding.status !== 'InProgress') return <Page title="조치 제출"><Copy>제출 가능한 작업이 없습니다.</Copy></Page>;
  const current = finding;
  const actor = user;

  function submit() {
    try {
      if (!photo || !note.trim()) throw new Error('조치 내용과 조치 후 사진이 필요합니다.');
      workflow.execute(actor, current.id, { kind: 'submit', note: [note.trim(), memo.trim()].filter(Boolean).join('\n'), photo });
      router.replace({ pathname: '/comparison', params: { id: current.id } });
    } catch (error) {
      setError(message(error));
    }
  }

  return <Page title="조치 제출">
    <Copy>{finding.location}</Copy>
    <Card>
      <Field label="수행한 조치" value={note} onChangeText={setNote} multiline />
      <PhotoField label="조치 후 사진" photo={photo} onChange={setPhoto} />
      <Field label="추가 메모 (선택)" value={memo} onChangeText={setMemo} multiline />
    </Card>
    {!!error && <Copy tone="error">{error}</Copy>}
    <Button title="조치 제출" onPress={submit} />
  </Page>;
}
