import { Directory, File, Paths } from 'expo-file-system';
import type { PhotoStore } from './photo-store';

let sequence = 0;

function extensionOf(uri: string): string {
  const name = uri.split(/[?#]/)[0].split('/').pop() ?? '';
  const match = /\.([A-Za-z0-9]{1,5})$/.exec(name);
  return match ? '.' + match[1].toLowerCase() : '.jpg';
}

function listFiles(directory: Directory): string[] {
  if (!directory.exists) return [];
  return directory.list().flatMap(entry => (entry instanceof Directory ? listFiles(entry) : [entry.uri]));
}

/** Copies photos into <document>/photos/<findingId>/ with expo-file-system (SDK 58 File/Directory API). */
export class AppPhotoStore implements PhotoStore {
  private readonly root = new Directory(Paths.document, 'photos');

  constructor() {
    if (!this.root.exists) this.root.create({ intermediates: true });
  }

  private isManaged(uri: string) {
    const root = this.root.uri.endsWith('/') ? this.root.uri : this.root.uri + '/';
    return uri.startsWith(root);
  }

  import(uri: string, findingId: string, name: string) {
    if (!uri || this.isManaged(uri)) return { uri, created: false };
    const folder = new Directory(this.root, findingId);
    if (!folder.exists) folder.create({ intermediates: true });
    const target = new File(folder, `${name}-${Date.now().toString(36)}${++sequence}${extensionOf(uri)}`);
    new File(uri).copySync(target);
    return { uri: target.uri, created: true };
  }

  remove(uri: string) {
    const file = new File(uri);
    if (file.exists) file.delete();
  }

  listManaged() {
    return listFiles(this.root);
  }
}
