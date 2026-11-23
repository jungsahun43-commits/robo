import { useState } from 'react';
import { Redirect, router, Stack } from 'expo-router';
import { Button, Card, Copy, Field, message, Page } from '@/components/safelog-ui';
import type { Role } from '@/contracts/types';
import { AuthService } from '@/features/auth/auth-service';
import { roleLabels } from '@/features/auth/role-labels';
import { createAccount, useSession } from '@/features/auth/session';

const roles: readonly Role[] = ['Inspector', 'Manager', 'Assignee'];

// Not listed in _layout.tsx, so no Stack.Protected guard applies: managers only,
// checked here (AuthService also refuses non-managers).
export default function AccountsPage() {
  const { user } = useSession();
  const [name, setName] = useState('');
  const [role, setRole] = useState<Role>('Inspector');
  const [loginId, setLoginId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [created, setCreated] = useState<{ name: string; role: Role; loginId: string }>();

  if (!user) return <Redirect href="/" />;
  if (user.role !== 'Manager') return <Redirect href="/home" />;

  async function submit() {
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      const profile = await createAccount(name, role, loginId, password);
      setCreated({ name: profile.name, role: profile.role, loginId: AuthService.normalizeLoginId(loginId) });
      setName('');
      setLoginId('');
      setPassword('');
    } catch (error) {
      setError(message(error));
    } finally {
      setBusy(false);
    }
  }

  return <Page title="계정 만들기">
    <Stack.Screen options={{ title: '계정 만들기' }} />
    {created ? <>
      <Card>
        <Copy tone="human">{created.name}({roleLabels[created.role]}) 계정을 만들었습니다.</Copy>
        <Copy>아이디 · {created.loginId}</Copy>
        <Copy>첫 로그인 때 비밀번호를 바꿔야 합니다.</Copy>
      </Card>
      <Button title="다른 계정 만들기" secondary onPress={() => setCreated(undefined)} />
      <Button title="홈으로" onPress={() => router.dismissTo('/home')} />
    </> : <>
      <Card>
        <Field label="이름" value={name} onChangeText={setName} />
        <Copy>역할 · {roleLabels[role]}</Copy>
        {roles.map(option => <Button key={option} title={roleLabels[option]} secondary={option !== role}
          disabled={busy} onPress={() => setRole(option)} />)}
      </Card>
      <Card>
        <Field label="아이디" autoCapitalize="none" autoCorrect={false} value={loginId} onChangeText={setLoginId} />
        <Copy>3~32자의 영문 소문자, 숫자, &quot;.&quot;, &quot;_&quot;, &quot;-&quot;</Copy>
        <Field label="임시 비밀번호" autoCapitalize="none" autoCorrect={false} value={password}
          onChangeText={setPassword} onSubmitEditing={submit} />
        <Copy>8자 이상 입력하세요. 새 사용자에게 전달하면, 첫 로그인 때 직접 바꾸게 됩니다.</Copy>
      </Card>
      {!!error && <Copy tone="error">{error}</Copy>}
      <Button title="계정 만들기" disabled={busy || !name.trim() || !loginId.trim() || !password} onPress={submit} />
    </>}
  </Page>;
}
