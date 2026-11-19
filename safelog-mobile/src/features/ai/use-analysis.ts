import { useEffect, useRef, useState } from 'react';
import type { Analysis } from '@/contracts/types';
import { message } from '@/components/safelog-ui';
import type { AiAnalyzer } from '@/contracts/interfaces';
import { mockAnalyzer, type MockResponse } from './mock';

export function useAnalysis() {
  const request = useRef<AbortController | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Analysis>();

  useEffect(() => () => request.current?.abort(), []);

  function cancel() {
    request.current?.abort();
    setBusy(false);
    setError('');
  }

  async function run(analyze: (analyzer: AiAnalyzer, signal: AbortSignal) => Promise<Analysis>, mode: MockResponse) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError('');
    try {
      const value = await analyze(mockAnalyzer(mode), controller.signal);
      if (!controller.signal.aborted) setResult(value);
    } catch (error) {
      if (!controller.signal.aborted) setError(message(error));
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return { busy, error, result, run, cancel };
}
