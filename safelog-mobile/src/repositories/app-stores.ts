import type { FindingRepository } from '@/contracts/interfaces';
import type { AccountStore } from '@/features/auth/account-store';
import { SqliteAccountStore } from './sqlite/account-store';
import { AppPhotoStore } from './sqlite/app-photo-store';
import { openExpoDatabase } from './sqlite/expo-database';
import { ensureSite, SqliteFindingRepository } from './sqlite/finding-repository';
import { migrate } from './sqlite/schema';

export interface AppStores {
  findings: FindingRepository;
  accounts: AccountStore;
}

/** Native (iOS/Android): one SQLite file for findings and credentials, photos under the app document folder. */
export function createAppStores(site: { id: string; name: string; address: string }): AppStores {
  const db = openExpoDatabase('safelog.db');
  migrate(db);
  ensureSite(db, site);
  const findings = new SqliteFindingRepository(db, new AppPhotoStore());
  try {
    findings.removeOrphanPhotos();
  } catch (error) {
    console.warn('[SafeLog] orphan photo clean-up failed', error);
  }
  return { findings, accounts: new SqliteAccountStore(db) };
}
