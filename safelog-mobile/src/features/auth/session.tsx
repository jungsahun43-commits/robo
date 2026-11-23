import { createContext, useCallback, useContext, useMemo, useState, useSyncExternalStore, type ReactNode } from 'react';
import type { Profile, Role } from '@/contracts/types';
import { createAppStores } from '@/repositories/app-stores';
import { visibleTo, WorkflowService } from '@/features/workflow/service';
import { AuthService, type LoginResult } from './auth-service';
import { measureScrypt } from './benchmark';
import { DEMO_MODE, DEMO_PASSWORD, demoAliases, profiles, site } from './demo';
import { PasswordHasher } from './password';
import { newProfileId, secureRandom } from './random';

const stores = createAppStores(site);
const repository = stores.findings;
export const workflow = new WorkflowService(repository, stores.accounts);

const peopleListeners = new Set<() => void>();
let peopleSnapshot = stores.accounts.listProfiles();
const subscribePeople = (listener: () => void) => {
  peopleListeners.add(listener);
  return () => { peopleListeners.delete(listener); };
};
const getPeople = () => peopleSnapshot;
const refreshPeople = () => {
  peopleSnapshot = stores.accounts.listProfiles();
  peopleListeners.forEach(listener => listener());
};

export const auth = new AuthService(stores.accounts, new PasswordHasher(secureRandom), {
  newProfileId,
  aliases: demoAliases,
});

// Demo logins are created once (existing ones are kept). Hashing takes a moment on first start.
const ready = DEMO_MODE ? auth.seedDemoAccounts(profiles, DEMO_PASSWORD) : Promise.resolve();
ready.then(refreshPeople).catch(error => console.warn('[SafeLog] demo account set-up failed', error));

/**
 * The person who last signed in (set even when they still must change their
 * password). changePassword / createAccount act as this person; logging out clears it.
 */
let signedIn: { profile: Profile; mustChangePassword: boolean } | null = null;

/** Signs in with the local credentials; throws AuthenticationError with one generic message. */
export async function signIn(loginId: string, password: string): Promise<LoginResult> {
  await ready;
  const result = await auth.login(loginId, password);
  signedIn = { ...result };
  return result;
}

/** Changes the signed-in user's password; afterwards they no longer have to change it. */
export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  if (!signedIn) throw new Error('로그인이 필요합니다.');
  await auth.changePassword(signedIn.profile.id, currentPassword, newPassword);
  signedIn = { ...signedIn, mustChangePassword: false };
}

/** Registers a new person as the signed-in user (AuthService allows managers only). */
export async function createAccount(name: string, role: Role, loginId: string, temporaryPassword: string): Promise<Profile> {
  if (!signedIn || signedIn.mustChangePassword) throw new Error('로그인이 필요합니다.');
  const profile = await auth.createAccount(signedIn.profile.id, name, role, loginId, temporaryPassword);
  refreshPeople();
  return profile;
}

// Device timing for choosing the scrypt cost: start with EXPO_PUBLIC_SAFELOG_BENCH=1 (dev only).
if (__DEV__ && process.env.EXPO_PUBLIC_SAFELOG_BENCH === '1') {
  ready.catch(() => undefined).then(() => measureScrypt(secureRandom))
    .then(results => console.log('[SafeLog] scrypt benchmark (r=8, p=1)\n' + results
      .map(r => `  logN=${r.logN} (${r.memoryMiB} MiB): avg ${r.averageMs} ms, runs ${r.runs.join('/')}`).join('\n')))
    .catch(error => console.warn('[SafeLog] scrypt benchmark failed', error));
}

interface SessionValue {
  /** Fully signed-in user; the router's guards open the app only for them. */
  user: Profile | null;
  setUser: (user: Profile | null) => void;
  /** Signed in but must change the temporary password first (user stays null until then). */
  pending: Profile | null;
  setPending: (profile: Profile | null) => void;
}

const Session = createContext<SessionValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [user, setUserState] = useState<Profile | null>(null);
  const [pending, setPending] = useState<Profile | null>(null);
  const setUser = useCallback((next: Profile | null) => {
    if (!next) signedIn = null;  // logout
    setPending(null);
    setUserState(next);
  }, []);
  return <Session.Provider value={{ user, setUser, pending, setPending }}>{children}</Session.Provider>;
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

const noPeople: readonly Profile[] = [];
/** Public profile names/roles only; credentials never enter the UI directory. */
export function useProfiles() {
  const { user } = useSession();
  const people = useSyncExternalStore(subscribePeople, getPeople, getPeople);
  return user ? people : noPeople;
}
