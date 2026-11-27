export const inspectionCategories = {
  ball_screw_pitting: { label: '볼스크루 표면 피팅', check: '나사 홈과 접촉면의 점상 박리·패임 확인' },
  facility_crack: { label: '시설 균열', check: '구조물 표면의 균열 위치·범위 확인' },
  conveyor_foreign: { label: '컨베이어 이물질', check: '벨트와 이송 구간에 이물질·적치물 유입 여부 확인' },
  robustad_pcb: { label: 'RobustAD PCB', check: 'PCB 표면·패턴·부품의 외관 이상 확인' },
  cable_exterior: { label: '케이블 외관', check: '피복의 벗겨짐·마모·눌림 등 외관 손상 확인' },
  belt_damage: { label: '컨베이어 벨트 손상', check: '벨트 표면의 찢김·마모 등 외관 손상 확인' },
  loco_splicing_connectors: { label: '전선 연결부 외관', check: '전선 연결부의 이탈·눌림·손상 여부 확인' },
  ad_screw: { label: '일반 나사 외관', check: '나사 머리와 나사산의 변형·손상 여부 확인' },
  ad_metal_nut: { label: '너트 외관', check: '너트의 외곽과 나사 구멍 손상 여부 확인' },
  mpdd_bracket_black: { label: '검정 브래킷 외관', check: '검정 브래킷의 변형·표면 손상 여부 확인' },
  mpdd_bracket_brown: { label: '갈색 브래킷 외관', check: '갈색 브래킷의 변형·표면 손상 여부 확인' },
  mpdd_bracket_white: { label: '흰색 브래킷 외관', check: '흰색 브래킷의 변형·표면 손상 여부 확인' },
  mpdd_connector: { label: '금속 연결부 외관', check: '금속 연결 부품의 외형 변형·표면 손상 여부 확인' },
  mpdd_metal_plate: { label: '금속판 외관', check: '금속판의 표면 흠집·변형 여부 확인' },
  mpdd_tubes: { label: '금속 튜브 외관', check: '금속 튜브의 눌림·변형·표면 손상 여부 확인' },
  ksdd2_surface: { label: '부품 표면 손상', check: '가공 부품 표면의 외관 손상 후보 확인' },
  pcb1: { label: 'PCB 외관 · VisA 1', check: 'VisA PCB1 유형의 기판·부품 외관 확인' },
  pcb2: { label: 'PCB 외관 · VisA 2', check: 'VisA PCB2 유형의 기판·부품 외관 확인' },
  pcb3: { label: 'PCB 외관 · VisA 3', check: 'VisA PCB3 유형의 기판·부품 외관 확인' },
  pcb4: { label: 'PCB 외관 · VisA 4', check: 'VisA PCB4 유형의 기판·부품 외관 확인' },
} as const;

export type InspectionCategory = keyof typeof inspectionCategories;
export const inspectionCategoryIds = Object.keys(inspectionCategories) as InspectionCategory[];
