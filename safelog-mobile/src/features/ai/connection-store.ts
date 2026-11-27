import { fixedCpuSettings } from './settings';

export interface AiConnection {
  readonly generalBaseUrl: string;
  readonly pcbBaseUrl: string;
  readonly demoToken?: string;
  readonly revision: number;
}

/** Runtime-only configuration: never write the demo token to SQLite, files, or logs. */
export function createAiConnectionStore(defaults = fixedCpuSettings) {
  let snapshot: AiConnection = Object.freeze({ ...defaults, revision: 0 });
  const listeners = new Set<() => void>();
  const publish = (value: Omit<AiConnection, 'revision'>) => {
    snapshot = Object.freeze({ ...value, revision: snapshot.revision + 1 });
    listeners.forEach(listener => listener());
  };
  return {
    getSnapshot: () => snapshot,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    configure(origin: string, token: string) {
      let url: URL;
      try { url = new URL(origin.trim()); } catch { throw new Error('HTTPS AI 서버 주소를 확인하세요.'); }
      if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
        throw new Error('경로·계정·쿼리 없는 HTTPS AI 서버 주소를 입력하세요.');
      }
      const secret = token.trim();
      if (!/^[A-Za-z0-9_-]{43,256}$/.test(secret)) throw new Error('발급받은 시연 연결 토큰을 확인하세요.');
      publish({ generalBaseUrl: url.origin + '/general', pcbBaseUrl: url.origin + '/pcb', demoToken: secret });
    },
    reset() { publish({ ...defaults }); },
  };
}

export const aiConnection = createAiConnectionStore();
