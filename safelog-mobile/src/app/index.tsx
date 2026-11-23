import { useState } from 'react';
import { Platform } from 'react-native';
import { Button, Card, Copy, Field, message, Page } from '@/components/safelog-ui';
import { signIn, useSession } from '@/features/auth/session';

export default function LoginPage() {
  const { setUser } = useSession();
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function login() {
    if (busy) return;
    setBusy(true);
    try {
      setUser((await signIn(id, password)).profile);
    } catch (error) {
      setError(message(error));
    } finally {
      setBusy(false);
    }
  }

  return <Page title="SafeLog AI 로그인">
    <Copy>{Platform.OS === 'web'
      ? 'Mock 개발 모드 · 앱을 종료하면 기록이 초기화됩니다.'
      : '기록은 이 기기에 저장됩니다.'}</Copy>
    <Card>
      <Field label="사용자 ID" autoCapitalize="none" autoCorrect={false} value={id} onChangeText={setId} />
      <Field label="비밀번호" secureTextEntry value={password} onChangeText={setPassword} onSubmitEditing={login} />
      {!!error && <Copy tone="error">{error}</Copy>}
      <Button title="로그인" disabled={busy || !id.trim() || !password} onPress={login} />
    </Card>
    <Card><Copy>{'김안전 · inspector-1\n박관리 · manager-1\n이조치 · assignee-1\n공통 비밀번호 · demo1234'}</Copy></Card>
  </Page>;
}
