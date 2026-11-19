import { createContext, useContext, useMemo, useState, useSyncExternalStore, type ReactNode } from 'react';
import type { Profile } from '@/contracts/types';
import { MemoryFindingRepository } from '@/repositories/memory';
import { visibleTo, WorkflowService } from '@/features/workflow/service';

const repository = new MemoryFindingRepository();
export const workflow = new WorkflowService(repository);

const Session = createContext<{ user: Profile | null; setUser: (user: Profile | null) => void } | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Profile | null>(null);
  return <Session.Provider value={{ user, setUser }}>{children}</Session.Provider>;
}

export function useSession() {
  const session = useContext(Session);
  if (!session) throw new Error('SessionProvider is required');
  return session;
}

export function useFindings() {
  const { user } = useSession();
  const snapshot = useSyncExternalStore(repository.subscribe, repository.snapshot, repository.snapshot);
  return useMemo(() => user ? snapshot.filter(finding => visibleTo(user, finding)) : [], [snapshot, user]);
}
