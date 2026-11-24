import { useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react';
import type { AiMode, Analysis, InspectionTask } from '@/contracts/types';
import { message } from '@/components/safelog-ui';
import type { AiAnalyzer } from '@/contracts/interfaces';
import { mockAnalyzer, type MockResponse } from './mock';
import { createFixedCpuAnalyzer } from './fixed-cpu';
import { readPhotoDataUrl } from './photo-data';
import { aiConnection } from './connection-store';
import { isFixedTask, isPcbTask } from '@/contracts/inspection';

export function useAnalysis(recordScope: string, aiMode: AiMode = 'mock', inspectionTask?: InspectionTask) {
  const connection = useSyncExternalStore(aiConnection.subscribe, aiConnection.getSnapshot, aiConnection.getSnapshot);
  const scope = recordScope + ':connection-' + connection.revision;
  const request = useRef<AbortController | null>(null);
  const latestScope = useRef(scope);
  const [state, setState] = useState<{ scope: string; busy: boolean; error: string; result?: Analysis }>({
    scope, busy: false, error: '',
  });

  useLayoutEffect(() => {
    latestScope.current = scope;
    return () => {
      request.current?.abort();
      request.current = null;
    };
  }, [scope]);

  function cancel() {
    request.current?.abort();
    request.current = null;
    setState({ scope, busy: false, error: '' });
  }

  async function run(analyze: (analyzer: AiAnalyzer, signal: AbortSignal) => Promise<Analysis>, mode: MockResponse) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setState({ scope, busy: true, error: '' });
    try {
      if (aiMode === 'manual') throw new Error('이 분야는 AI 없이 직접 검토합니다.');
      if (aiMode === 'fixed-cpu' && !isFixedTask(inspectionTask)) throw new Error('AI가 지원하는 점검 분야를 선택하세요.');
      const analyzer = aiMode === 'fixed-cpu' ? createFixedCpuAnalyzer({
        ...connection, task: inspectionTask!,
        readPhoto: (photo, signal) => readPhotoDataUrl(photo, signal, isPcbTask(inspectionTask) ? 40_000_000 : 12_000_000),
      }) : mockAnalyzer(mode);
      const value = await analyze(analyzer, controller.signal);
      if (request.current === controller && latestScope.current === scope && !controller.signal.aborted) {
        setState({ scope, busy: false, error: '', result: value });
      }
    } catch (error) {
      if (request.current === controller && latestScope.current === scope && !controller.signal.aborted) {
        setState({ scope, busy: false, error: message(error) });
      }
    } finally {
      if (request.current === controller) request.current = null;
    }
  }

  const isCurrent = (signal: AbortSignal) => request.current?.signal === signal && latestScope.current === scope && !signal.aborted;
  const current = state.scope === scope ? state : { busy: false, error: '', result: undefined };
  return { ...current, run, cancel, isCurrent };
}
