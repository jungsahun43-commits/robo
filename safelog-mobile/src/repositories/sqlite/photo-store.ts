/**
 * Where photo files live. The repository stores only the returned URI in SQLite.
 * The app uses AppPhotoStore (expo-file-system); tests use their own fakes.
 */
export interface PhotoStore {
  /**
   * Returns a URI inside app-managed storage for `uri`, copying the file there
   * when it is not already managed. `created` is true when a new file was written,
   * so the caller can delete it if the database write fails.
   */
  import(uri: string, findingId: string, name: string): { uri: string; created: boolean };
  /** Deletes a managed file; ignores files that no longer exist. */
  remove(uri: string): void;
  /** Every file currently in managed storage (for orphan clean-up). */
  listManaged(): string[];
}

/** Keeps URIs as they are (no copying). For tests and platforms without a file system. */
export const passthroughPhotoStore: PhotoStore = {
  import: uri => ({ uri, created: false }),
  remove: () => {},
  listManaged: () => [],
};
