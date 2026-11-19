import { useLocalSearchParams } from 'expo-router';
import { useFindings, useSession } from '@/features/auth/session';

export function useFinding() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const { user } = useSession();
  const records = useFindings();
  const finding = records.find(item => item.id === id);
  return { user, finding };
}
