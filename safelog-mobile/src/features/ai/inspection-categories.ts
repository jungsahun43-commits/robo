export const inspectionCategories = {
  ball_screw_pitting: { label: '볼스크루 표면 피팅', check: '나사 홈과 접촉면의 점상 박리·패임 확인' },
  facility_crack: { label: '시설 균열', check: '구조물 표면의 균열 위치·범위 확인' },
  conveyor_foreign: { label: '컨베이어 이물질', check: '벨트와 이송 구간에 이물질·적치물 유입 여부 확인' },
  robustad_pcb: { label: 'RobustAD PCB', check: 'PCB 표면·패턴·부품의 외관 이상 확인' },
  cable_exterior: { label: '케이블 외관', check: '피복의 벗겨짐·마모·눌림 등 외관 손상 확인' },
} as const;

export type InspectionCategory = keyof typeof inspectionCategories;
export const inspectionCategoryIds = Object.keys(inspectionCategories) as InspectionCategory[];
