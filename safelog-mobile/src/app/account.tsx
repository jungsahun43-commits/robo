import { router } from 'expo-router';
import { Platform } from 'react-native';
import { Button, Card, Copy, Page } from '@/components/safelog-ui';
import { site } from '@/features/auth/demo';
import { roleLabels } from '@/features/auth/role-labels';
import { useSession } from '@/features/auth/session';

export default function AccountPage() {
  const { user, setUser } = useSession();
  if (!user) return null;

  return <Page title="내 계정">
    <Card>
      <Copy>{user.name} · {roleLabels[user.role]}</Copy>
      <Copy>{user.id}</Copy>
      <Copy>{site.name}</Copy>
      <Button title="현장 정보 보기" secondary onPress={() => router.push('/facilities')} />
      <Button title="비밀번호 변경" secondary onPress={() => router.push('/password')} />
    </Card>
    <Card>
      <Copy>시연 모드</Copy>
      <Copy>{Platform.OS === 'web'
        ? 'AI 응답은 시연용이며, 점검 기록은 앱을 종료하면 초기화됩니다.'
        : 'AI 응답은 시연용이며, 점검 기록은 이 기기에 저장됩니다.'}</Copy>
    </Card>
    <Button title="로그아웃" secondary onPress={() => setUser(null)} />
  </Page>;
}
