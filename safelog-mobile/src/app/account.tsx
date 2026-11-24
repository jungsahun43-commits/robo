import { useState } from 'react';
import { router } from 'expo-router';
import { Alert, Platform } from 'react-native';
import { Button, Card, Copy, message, Page } from '@/components/safelog-ui';
import { DEMO_MODE, site } from '@/features/auth/demo';
import { roleLabels } from '@/features/auth/role-labels';
import { resetDemoData, useSession } from '@/features/auth/session';
import { AiConnectionPanel } from '@/features/ai/connection-panel';

const RESET_TITLE = '시연 데이터 초기화';
const RESET_MESSAGE = '모든 점검 기록과 사진, 시연 계정이 아닌 계정을 삭제하고 시연 계정 비밀번호를 demo1234로 되돌립니다. 되돌릴 수 없습니다.';

/** Asks before a destructive action. Alert.alert does nothing on react-native-web, so the web uses window.confirm. */
function confirmDestructive(onConfirm: () => void) {
  if (Platform.OS === 'web') {
    const confirm = (globalThis as { confirm?: (text: string) => boolean }).confirm;
    if (confirm?.(`${RESET_TITLE}\n\n${RESET_MESSAGE}`)) onConfirm();
    return;
  }
  Alert.alert(RESET_TITLE, RESET_MESSAGE, [
    { text: '취소', style: 'cancel' },
    { text: '초기화', style: 'destructive', onPress: onConfirm },
  ]);
}

export default function AccountPage() {
  const { user, setUser } = useSession();
  const [resetting, setResetting] = useState(false);
  const [resetNotice, setResetNotice] = useState('');
  const [resetError, setResetError] = useState('');
  if (!user) return null;

  async function reset() {
    setResetting(true);
    setResetNotice('');
    setResetError('');
    try {
      const stillSignedIn = await resetDemoData();
      if (!stillSignedIn) setUser(null);
      else setResetNotice('시연 데이터를 초기화했습니다. 시연 계정 비밀번호는 demo1234입니다.');
    } catch (error) {
      setResetError(message(error));
    } finally {
      setResetting(false);
    }
  }

  return <Page title="내 계정">
    <Card>
      <Copy>{user.name} · {roleLabels[user.role]}</Copy>
      <Copy>{user.id}</Copy>
      <Copy>{site.name}</Copy>
      <Button title="현장 정보 보기" secondary onPress={() => router.push('/facilities')} />
      <Button title="비밀번호 변경" secondary onPress={() => router.push('/password')} />
    </Card>
    <Card>
      <Copy>AI 분석과 기록 보관</Copy>
      <Copy>{Platform.OS === 'web'
        ? '기존 고정 AI·모의 시연·직접 검토를 점검별로 선택합니다. 웹의 기록과 계정은 새로고침하면 초기화됩니다.'
        : '기존 고정 AI·모의 시연·직접 검토를 점검별로 선택합니다. 점검 사진과 분석·검토 기록은 이 기기에 저장됩니다.'}</Copy>
      <Copy>전체 AI 20항목을 지원하며, 발표는 볼스크루·균열·컨베이어 이물질·PCB·케이블로 구성합니다.</Copy>
    </Card>
    <AiConnectionPanel />
    {DEMO_MODE && user.role === 'Manager' && <Card>
      <Copy>{RESET_TITLE}</Copy>
      <Copy>시연 전에 테스트 기록과 임시 계정을 지웁니다. 시연 계정 3개만 남습니다.</Copy>
      {!!resetNotice && <Copy tone="human">{resetNotice}</Copy>}
      {!!resetError && <Copy tone="error">{resetError}</Copy>}
      <Button title={resetting ? '초기화 중…' : RESET_TITLE} disabled={resetting}
        onPress={() => confirmDestructive(() => { void reset(); })} />
    </Card>}
    <Button title="로그아웃" secondary onPress={() => setUser(null)} />
  </Page>;
}
