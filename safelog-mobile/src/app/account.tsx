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
      <Copy>AI 분석과 기록 보관</Copy>
      <Copy>{Platform.OS === 'web'
        ? '기존 고정 AI·모의 시연·직접 검토를 점검별로 선택합니다. 웹의 기록과 계정은 새로고침하면 초기화됩니다.'
        : '기존 고정 AI·모의 시연·직접 검토를 점검별로 선택합니다. 점검 사진과 분석·검토 기록은 이 기기에 저장됩니다.'}</Copy>
      <Copy>전체 AI 20항목을 지원하며, 발표는 볼스크루·균열·컨베이어 이물질·PCB·케이블로 구성합니다.</Copy>
    </Card>
    <Button title="로그아웃" secondary onPress={() => setUser(null)} />
  </Page>;
}
