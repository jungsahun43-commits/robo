import { useEffect, useState } from 'react';
import { Redirect, router, Stack } from 'expo-router';
import { Button, Card, Copy, Field, message, Page } from '@/components/safelog-ui';
import { changePassword, useSession } from '@/features/auth/session';

// Not listed in _layout.tsx, so no Stack.Protected guard applies: this screen
// checks the session itself. While `pending` is set (temporary password),
// `user` is still null and every guarded screen stays closed.
export default function PasswordPage() {
  const { user, pending, setUser } = useSession();
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [leaving, setLeaving] = useState(false);

  // Navigate once the guards have re-rendered with the signed-in user.
  useEffect(() => {
    if (leaving && user) router.dismissTo('/home');
  }, [leaving, user]);

  const person = user ?? pending;
  if (!person) return <Redirect href="/" />;
  const mismatch = !!confirm && next !== confirm;

  async function submit() {
    if (busy) return;
    if (next !== confirm) {
      setError('새 비밀번호와 확인이 일치하지 않습니다.');
      return;
    }
    setBusy(true);
    setError('');
    try {
      await changePassword(current, next);
      setDone(true);
    } catch (error) {
      setError(message(error));
    } finally {
      setBusy(false);
    }
  }

  function goHome() {
    if (!user && pending) setUser(pending);
    setLeaving(true);
  }

  return <Page title="비밀번호 변경">
    <Stack.Screen options={{ title: '비밀번호 변경', headerBackVisible: !!user }} />
    {done ? <>
      <Card>
        <Copy tone="human">비밀번호를 변경했습니다.</Copy>
        <Copy>다음 로그인부터 새 비밀번호를 사용하세요.</Copy>
      </Card>
      <Button title="홈으로" disabled={leaving} onPress={goHome} />
    </> : <>
      {!!pending && !user && <Card>
        <Copy>{pending.name}님, 처음 로그인했습니다.</Copy>
        <Copy>임시 비밀번호를 새 비밀번호로 바꿔야 앱을 사용할 수 있습니다.</Copy>
      </Card>}
      <Card>
        <Field label="현재 비밀번호" secureTextEntry autoCapitalize="none" autoCorrect={false}
          value={current} onChangeText={setCurrent} />
        <Field label="새 비밀번호" secureTextEntry autoCapitalize="none" autoCorrect={false}
          value={next} onChangeText={setNext} />
        <Field label="새 비밀번호 확인" secureTextEntry autoCapitalize="none" autoCorrect={false}
          value={confirm} onChangeText={setConfirm} onSubmitEditing={submit} />
        <Copy>8자 이상 입력하세요.</Copy>
        {mismatch && <Copy tone="error">새 비밀번호와 확인이 일치하지 않습니다.</Copy>}
        {!!error && <Copy tone="error">{error}</Copy>}
        <Button title="비밀번호 변경" disabled={busy || !current || !next || !confirm || mismatch} onPress={submit} />
      </Card>
      {!user && <Button title="로그아웃" secondary disabled={busy} onPress={() => setUser(null)} />}
    </>}
  </Page>;
}
