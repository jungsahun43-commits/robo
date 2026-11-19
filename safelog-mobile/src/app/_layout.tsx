import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { palette } from '@/constants/safelog-theme';
import { SessionProvider, useSession } from '@/features/auth/session';

function Navigation() {
  const { user } = useSession();
  return <>
    <StatusBar style="dark" />
    <Stack screenOptions={{ headerTintColor: palette.ink, headerBackButtonDisplayMode: 'minimal' }}>
      <Stack.Protected guard={!user}>
        <Stack.Screen name="index" options={{ title: 'SafeLog' }} />
      </Stack.Protected>
      <Stack.Protected guard={!!user}>
        <Stack.Screen name="home" options={{ headerShown: false }} />
        <Stack.Screen name="account" options={{ title: '계정' }} />
        <Stack.Screen name="facilities" options={{ title: '시설물현황' }} />
        <Stack.Screen name="tasks" options={{ title: '조치 관리' }} />
        <Stack.Screen name="comparison" options={{ title: '전후 비교' }} />
        <Stack.Protected guard={user?.role === 'Inspector'}>
          <Stack.Screen name="analysis" options={{ title: 'AI 위험 분석' }} />
          <Stack.Screen name="capture" options={{ title: '안전점검 등록' }} />
          <Stack.Screen name="hazard" options={{ title: 'AI 위험 검토' }} />
          <Stack.Screen name="saved" options={{ title: '점검 등록' }} />
        </Stack.Protected>
        <Stack.Protected guard={user?.role === 'Assignee'}>
          <Stack.Screen name="action" options={{ title: '조치 제출' }} />
        </Stack.Protected>
        <Stack.Protected guard={user?.role !== 'Assignee'}>
          <Stack.Screen name="reports" options={{ title: '점검 보고서' }} />
          <Stack.Screen name="report" options={{ title: '보고서' }} />
        </Stack.Protected>
      </Stack.Protected>
      <Stack.Protected guard={false}><Stack.Screen name="explore" /></Stack.Protected>
    </Stack>
  </>;
}

export default function RootLayout() {
  return <SessionProvider><Navigation /></SessionProvider>;
}
