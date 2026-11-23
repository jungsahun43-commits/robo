import { createContext, useContext, useMemo, useState, useSyncExternalStore, type ReactNode } from 'react';
import type { Profile } from '@/contracts/types';
import { createAppStores } from '@/repositories/app-stores';
import { visibleTo, WorkflowService } from '@/features/workflow/service';
import { AuthService } from './auth-service';
import { measureScrypt } from './benchmark';
import { DEMO_MODE, DEMO_PASSWORD, demoAliases, profiles, site } from './demo';
import { PasswordHasher } from './password';
import { newProfileId, secureRandom } from './random';

const stores = createAppStores(site);
const repository = stores.findings;
export const workflow = new WorkflowService(repository);

export const auth = new AuthService(stores.accounts, new PasswordHasher(secureRandom), {
  newProfileId,
  aliases: demoAliases,
});

// Demo logins are created once (existing ones are kept). Hashing takes a moment on first start.
const ready = DEMO_MODE ? auth.seedDemoAccounts(profiles, DEMO_PASSWORD) : Promise.resolve();
ready.catch(error => console.warn('[SafeLog] demo account set-up failed', error));

/** Signs in with the local credentials; throws AuthenticationError with one generic message. */
export async function signIn(loginId: string, password: string): Promise<Profile> {
  await ready;
  return (await auth.login(loginId, password)).profile;
}

// Device timing for choosing the scrypt cost: start with EXPO_PUBLIC_SAFELOG_BENCH=1 (dev only).
if (__DEV__ && process.env.EXPO_PUBLIC_SAFELOG_BENCH === '1') {
  ready.catch(() => undefined).then(() => measureScrypt(secureRandom))
    .then(results => console.log('[SafeLog] scrypt benchmark (r=8, p=1)\n' + results
      .map(r => `  logN=${r.logN} (${r.memoryMiB} MiB): avg ${r.averageMs} ms, runs ${r.runs.join('/')}`).join('\n')))
    .catch(error => console.warn('[SafeLog] scrypt benchmark failed', error));
}

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
