import { MemoryAccountStore } from '@/features/auth/account-store';
import type { AppStores } from './app-stores';
import { MemoryFindingRepository } from './memory';

/**
 * Web: expo-file-system has no web support and expo-sqlite on web needs extra
 * wasm/COOP/COEP set-up, so records and accounts stay in memory (reset on reload).
 * The same AuthService runs on top of MemoryAccountStore.
 */
export function createAppStores(): AppStores {
  const findings = new MemoryFindingRepository();
  const accounts = new MemoryAccountStore();
  return {
    findings,
    accounts,
    demo: {
      resetDemoData(keep, credentials) {
        accounts.replaceAccounts(keep, credentials);
        findings.clear();  // no photo files are copied on web
      },
    },
  };
}
