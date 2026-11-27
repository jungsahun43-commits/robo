import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Button, Card, Copy, Field, message } from '@/components/safelog-ui';
import { aiConnection } from './connection-store';
import { createFixedCpuAnalyzer } from './fixed-cpu';

export function AiConnectionPanel() {
  const connection = useSyncExternalStore(aiConnection.subscribe, aiConnection.getSnapshot, aiConnection.getSnapshot);
  const [address, setAddress] = useState(() => connection.demoToken ? new URL(connection.generalBaseUrl).origin : '');
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);

  function save() {
    request.current?.abort();
    setBusy(false);
    setStatus('');
    setError('');
    try {
      aiConnection.configure(address, token);
      setToken('');
      setStatus('연결 정보를 적용했습니다. 모델 연결 확인을 눌러 주세요.');
    } catch (error) { setError(message(error)); }
  }

  async function check() {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const active = aiConnection.getSnapshot();
    setBusy(true);
    setStatus('');
    setError('');
    try {
      for (const task of ['facility_surface', 'robustad_pcb'] as const) {
        const analyzer = createFixedCpuAnalyzer({ ...active, task, timeoutMs: 15_000,
          readPhoto: async () => { throw new Error('연결 확인은 사진을 전송하지 않습니다.'); } });
        await analyzer.checkConnection(controller.signal);
      }
      if (!controller.signal.aborted && aiConnection.getSnapshot().revision === active.revision) {
        setStatus('기존 고정 모델 2개의 준비 상태와 식별값을 확인했습니다. 이제 사진 분석을 진행하세요.');
      }
    } catch (error) {
      if (!controller.signal.aborted) setError(message(error));
    } finally {
      if (request.current === controller) { request.current = null; setBusy(false); }
    }
  }

  function clear() {
    request.current?.abort();
    request.current = null;
    aiConnection.reset();
    setAddress(''); setToken(''); setBusy(false); setError('');
    setStatus('시연 연결 정보를 지웠습니다. 기본 로컬 주소로 돌아갑니다.');
  }

  return <Card>
    <Copy>아이폰·외부 기기 AI 시연 연결</Copy>
    <Copy>전달받은 HTTPS 주소와 시연 토큰을 입력하세요. 토큰은 앱 실행 중에만 보관하며 앱 종료·Reload 후 다시 입력합니다.</Copy>
    <Field label="HTTPS AI 서버 주소" value={address} onChangeText={setAddress} editable={!busy}
      placeholder="https://example.trycloudflare.com" autoCapitalize="none" autoCorrect={false}
      autoComplete="off" keyboardType="url" maxLength={2048} />
    <Field label="시연 연결 토큰" value={token} onChangeText={setToken} editable={!busy}
      secureTextEntry autoCapitalize="none" autoCorrect={false} autoComplete="off" maxLength={256} />
    <Button title="시연 연결 적용" onPress={save} disabled={busy} />
    <Button title={busy ? '모델 연결 확인 중…' : '모델 연결 확인'} onPress={() => { void check(); }} disabled={busy} secondary />
    <Button title="시연 연결 지우기" onPress={clear} secondary />
    <Copy>{connection.demoToken ? '시연 토큰이 적용되어 있습니다.' : '기본 로컬 연결을 사용 중입니다.'}</Copy>
    {!!status && <Copy>{status}</Copy>}
    {!!error && <Copy tone="error">{error}</Copy>}
  </Card>;
}
