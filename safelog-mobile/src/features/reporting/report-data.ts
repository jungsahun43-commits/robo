import type { Finding, Profile } from '@/contracts/types';
import { profiles, site } from '@/features/auth/demo';
import { taskLabel } from '@/features/ai/tasks';

export function reportFor(finding: Finding, people: readonly Profile[] = profiles) {
  const name = (id?: string) => people.find(profile => profile.id === id)?.name ?? id ?? '미배정';
  return {
    header: [site.name, site.address, '점검자 · ' + name(finding.inspectorId), '점검 일시 · ' + finding.createdAt],
    record: [
      '기록 ID · ' + finding.id,
      '점검 ID · ' + finding.inspectionId,
      '장소 · ' + finding.location,
      '발견 내용 · ' + finding.description,
      '권장 조치 · ' + finding.action,
      '담당자 · ' + name(finding.assigneeId),
      '위험 등급 · ' + (finding.risk ?? '미지정'),
      '작업 상태 · ' + finding.status,
      '분석 방식 · ' + (finding.aiMode === 'fixed-cpu' ? '고정 CPU 모델 / ' + taskLabel(finding.inspectionTask)
        : finding.aiMode === 'manual' ? 'AI 없이 직접 검토 / ' + taskLabel(finding.inspectionTask) : '모의 시연(Mock)'),
    ],
    photos: [{ label: '조치 전 사진', photo: finding.photo }, ...finding.after.map((photo, index) => ({
      label: '조치 후 사진 ' + (index + 1), photo,
    }))],
    ai: finding.analyses.map(item => ({
      id: item.id,
      text: [item.type + ' · ' + item.model + ' / ' + item.prompt, item.assessment, item.action,
        item.source === 'fixed-cpu' ? '고정 CPU 모델 · ' + taskLabel(item.inspectionTask) : '모의 시연(Mock)',
        item.candidateDetected === undefined ? '' : '외관 후보 · ' + (item.candidateDetected ? '감지' : '미검출(안전 보증 아님)'),
        item.anomalyScore !== undefined ? 'AI 참고 점수(확률 아님) · ' + item.anomalyScore + ' / 기준 ' + item.threshold
          : item.confidence !== undefined && !(item.source === 'fixed-cpu' && item.type === 'AfterComparison')
            ? 'AI 참고 점수(확률 아님) · ' + item.confidence : '',
        item.modelSha256 ? '모델 SHA256 · ' + item.modelSha256 : '',
        item.inputSha256 ? '입력 SHA256 · ' + item.inputSha256 : '',
        '검토 · ' + item.decision + (item.reviewer ? ' · ' + name(item.reviewer) : ''), item.at].filter(Boolean).join('\n'),
    })),
    logs: finding.logs.map(item => [item.action + ' · ' + name(item.actor), item.note, item.at].filter(Boolean).join('\n')),
  };
}
