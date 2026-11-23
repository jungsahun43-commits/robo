import type { InspectionTask } from '@/contracts/types';

export interface TaskOption {
  id: InspectionTask;
  label: string;
  group: string;
  scope: string;
  evaluation: string;
  photoRates?: { fp: number; normal: number; fn: number; defect: number; source: string };
}
const rates = (fp: number, normal: number, fn: number, defect: number, source = '알려진 공개 평가를 재사용한 개발 결과') =>
  ({ fp, normal, fn, defect, source });
export const inspectionTasks: TaskOption[] = [
  { id: 'facility_surface', label: '시설 표면 손상', group: '시설·컨베이어', scope: '균열·박락·녹 얼룩·철근 노출·젖음·백화·표면 공동. 아래 수치는 DamSegment 균열 항목만 해당합니다.', evaluation: '균열 항목 연구 평가', photoRates: rates(2,176,2,248,'DamSegment 균열 항목의 알려진 평가') },
  { id: 'belt_damage', label: '컨베이어 벨트 손상', group: '시설·컨베이어', scope: '벨트 표면 손상 후보. 정상 벨트 사진 오탐은 미측정입니다.', evaluation: '정상 사진 오탐 미측정' },
  { id: 'conveyor_foreign', label: '컨베이어 이물질', group: '시설·컨베이어', scope: '철광석 벨트 근접 사진의 이물질 후보입니다.', evaluation: '사진 유무 연구 평가', photoRates: rates(8,256,3,192,'알려진 촬영 묶음의 이물질 사진 평가') },
  { id: 'ad_cable', label: '케이블 외관', group: '전기·배선', scope: '등록된 케이블 근접 외관. 전기적 작동은 판단하지 않습니다.', evaluation: '사진 연구 평가', photoRates: rates(6,58,7,92) },
  { id: 'loco_splicing_connectors', label: '전선 연결부 외관', group: '전기·배선', scope: '등록된 연결부 사진 문맥과 외관. 미탐이 높아 직접 확인이 필요합니다.', evaluation: '개선 필요', photoRates: rates(23,119,77,193) },
  { id: 'ad_screw', label: '일반 나사 외관', group: '금속·기계 부품', scope: '등록된 방향·근접 외관. 나사산 옆면 미탐 13/23이 남아 있습니다.', evaluation: '유형별 성능 차이', photoRates: rates(2,41,15,119) },
  { id: 'ad_metal_nut', label: '너트 외관', group: '금속·기계 부품', scope: '등록된 금속 너트 근접 외관입니다.', evaluation: '사진 연구 평가', photoRates: rates(3,22,11,93) },
  { id: 'mpdd_bracket_black', label: '검정 브래킷 외관', group: '금속·기계 부품', scope: '지정된 검정 브래킷 사진만 해당하며 미탐이 높습니다.', evaluation: '개선 필요', photoRates: rates(4,32,30,47) },
  { id: 'mpdd_bracket_brown', label: '갈색 브래킷 외관', group: '금속·기계 부품', scope: '지정된 갈색 브래킷 근접 외관입니다.', evaluation: '사진 연구 평가', photoRates: rates(5,26,2,51) },
  { id: 'mpdd_bracket_white', label: '흰색 브래킷 외관', group: '금속·기계 부품', scope: '지정된 흰색 브래킷 근접 외관입니다.', evaluation: '사진 연구 평가', photoRates: rates(6,30,2,30) },
  { id: 'mpdd_connector', label: '금속 연결부 외관', group: '금속·기계 부품', scope: '손상 평가 사진이 14장으로 고정 최소 20장에 미달합니다.', evaluation: '평가 표본 부족', photoRates: rates(4,30,10,14) },
  { id: 'mpdd_metal_plate', label: '금속판 외관', group: '금속·기계 부품', scope: '지정된 금속판 외관. 낮은 오탐과 별개로 사진 미탐은 18.31%입니다.', evaluation: '사진 연구 평가', photoRates: rates(1,26,13,71) },
  { id: 'mpdd_tubes', label: '금속 튜브 외관', group: '금속·기계 부품', scope: '지정된 튜브 외관만 해당하며 미탐이 높습니다.', evaluation: '개선 필요', photoRates: rates(1,32,47,69) },
  { id: 'ksdd2_surface', label: '부품 표면 손상', group: '금속·기계 부품', scope: 'KSDD2와 같은 부품 표면 사진의 후보 판정. 사진 성능과 위치 성능은 별도입니다.', evaluation: '사진 연구 평가', photoRates: rates(110,894,5,110) },
  { id: 'ball_screw_pitting', label: '볼스크루 표면 피팅', group: '금속·기계 부품', scope: '볼스크루 표면을 잘라낸 크롭 사진이 필요합니다. 전체 부품·내부 기계 고장은 판단하지 않으며 촬영 설비별 독립성은 미검증입니다.', evaluation: '크롭 사진 연구 평가', photoRates: rates(14,1643,10,1593,'그룹화된 예약 크롭 사진 평가·물리적 스핀들 독립성 미검증') },
  { id: 'pcb1', label: 'PCB 외관 · VisA 1', group: '회로기판', scope: 'VisA PCB1과 같은 등록된 유형의 근접 사진만 해당합니다.', evaluation: '유형별 연구 평가', photoRates: rates(10,100,5,100,'VisA PCB1의 재사용 평가') },
  { id: 'pcb2', label: 'PCB 외관 · VisA 2', group: '회로기판', scope: 'VisA PCB2와 같은 등록된 유형의 근접 사진만 해당합니다.', evaluation: '유형별 연구 평가', photoRates: rates(2,100,19,100,'VisA PCB2의 재사용 평가') },
  { id: 'pcb3', label: 'PCB 외관 · VisA 3', group: '회로기판', scope: 'VisA PCB3과 같은 등록된 유형만 해당하며 사진 미탐은 31%입니다.', evaluation: '개선 필요', photoRates: rates(5,101,31,100,'VisA PCB3의 재사용 평가') },
  { id: 'pcb4', label: 'PCB 외관 · VisA 4', group: '회로기판', scope: 'VisA PCB4와 같은 등록된 유형의 근접 사진만 해당합니다. 평가에서 미탐 0/100은 새 사진 무미탐 보증이 아닙니다.', evaluation: '유형별 연구 평가', photoRates: rates(9,101,0,100,'VisA PCB4의 재사용 평가') },
  { id: 'robustad_pcb', label: 'PCB 외관 · RobustAD', group: '회로기판', scope: 'RobustAD와 같은 등록된 유형의 근접 사진만 해당합니다. 내부 전기 고장은 판단하지 않습니다.', evaluation: '유형별 연구 평가', photoRates: rates(4,120,1,75,'RobustAD TEST0 최초 평가·현장 독립성 미검증') },
  { id: 'manual_other', label: '기타 분야 · 직접 검토', group: '기타', scope: '기어·모터·보호구·추가 시설물 등 현재 두 고정 모델에 연결되지 않은 분야는 AI 판정 없이 기록하고 직접 검토합니다.', evaluation: '현재 AI 미연결' },
];
export const taskOption = (task?: InspectionTask) => inspectionTasks.find(item => item.id === task);
export const taskLabel = (task?: InspectionTask) => taskOption(task)?.label ?? '모의 점검';
export const taskGroups = [...new Set(inspectionTasks.map(task => task.group))];
// Published photo-rate ordering stays separate from the user's presentation selection.
export const researchTopTasks = inspectionTasks.filter(task => task.photoRates && task.photoRates.normal >= 20 && task.photoRates.defect >= 20)
  .sort((a,b) => Math.max(a.photoRates!.fp/a.photoRates!.normal,a.photoRates!.fn/a.photoRates!.defect)
    - Math.max(b.photoRates!.fp/b.photoRates!.normal,b.photoRates!.fn/b.photoRates!.defect)).slice(0,5);
const demoTaskIds = ['ball_screw_pitting', 'facility_surface', 'conveyor_foreign', 'robustad_pcb', 'ad_cable'] as const;
export const demoTasks = demoTaskIds.map(id => taskOption(id)!);
export const demoSelectionNote = '볼스크루·균열·이물질·PCB·케이블로 선정한 발표 구성입니다. 연구 수치와 분야 구성을 반영했으며 공통 현장 성능 순위는 아닙니다.';
export const isDemoTask = (task?: InspectionTask) => demoTasks.some(item => item.id === task);
export const photoRatesLabel = (task?: InspectionTask) => {
  const value = taskOption(task)?.photoRates;
  return value ? `연구 사진 오탐 ${(100*value.fp/value.normal).toFixed(2)}% (${value.fp}/${value.normal}) · 미탐 ${(100*value.fn/value.defect).toFixed(2)}% (${value.fn}/${value.defect})` : '사진 오탐·미탐 평가가 충분하지 않습니다.';
};
