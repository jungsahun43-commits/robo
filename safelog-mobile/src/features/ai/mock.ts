import type { AiAnalyzer } from '@/contracts/interfaces';
import type { Suggestion } from '@/contracts/types';

export type MockResponse = '정상' | '분석 실패' | '읽을 수 없는 결과';

export function mockAnalyzer(mode: MockResponse): AiAnalyzer {
  async function respond(value: Suggestion, signal: AbortSignal) {
    await new Promise<void>((resolve, reject) => {
      const cancel = () => {
        clearTimeout(timer);
        reject(new Error('분석을 취소했습니다.'));
      };
      const timer = setTimeout(() => {
        signal.removeEventListener('abort', cancel);
        resolve();
      }, 450);
      if (signal.aborted) cancel();
      else signal.addEventListener('abort', cancel, { once: true });
    });
    if (mode !== '정상') throw new Error(mode === '분석 실패' ? 'Mock 분석 서비스 오류' : 'Mock 응답을 읽을 수 없습니다.');
    return value;
  }

  const metadata = { model: 'safelog-mobile-mock', prompt: 'demo-v1', confidence: 0.82, source: 'mock' as const };
  return {
    hazard: (_before, signal) => respond({
      ...metadata, category: '통로 장애물', risk: 3,
      assessment: '통로에 적치물이 있어 이동을 방해할 가능성이 있습니다.',
      action: '적치물을 지정 보관구역으로 옮기고 통로 폭을 확인하세요.',
    }, signal),
    compare: (_before, _after, signal) => respond({
      ...metadata, assessment: '적치가 개선되었을 가능성이 있습니다. 사진 밖의 잔여 위험은 확인이 필요합니다.',
      action: '점검자가 직접 통로와 조치 내용을 확인하세요.',
    }, signal),
  };
}
