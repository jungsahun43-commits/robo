export const equipmentTaskIds = [
  'ad_cable', 'ad_screw', 'ad_metal_nut', 'loco_splicing_connectors',
  'mpdd_bracket_black', 'mpdd_bracket_brown', 'mpdd_bracket_white',
  'mpdd_connector', 'mpdd_metal_plate', 'mpdd_tubes', 'ksdd2_surface', 'ball_screw_pitting',
] as const;
export const pcbTaskIds = ['pcb1', 'pcb2', 'pcb3', 'pcb4', 'robustad_pcb'] as const;
export const fixedTaskIds = ['facility_surface', 'belt_damage', 'conveyor_foreign', ...equipmentTaskIds, ...pcbTaskIds] as const;
export type FixedInspectionTask = typeof fixedTaskIds[number];
export type InspectionTask = FixedInspectionTask | 'manual_other';
export const isFixedTask = (value: unknown): value is FixedInspectionTask =>
  typeof value === 'string' && (fixedTaskIds as readonly string[]).includes(value);
export const isPcbTask = (value: unknown): value is typeof pcbTaskIds[number] =>
  typeof value === 'string' && (pcbTaskIds as readonly string[]).includes(value);
export const isEquipmentTask = (value: unknown): value is typeof equipmentTaskIds[number] =>
  typeof value === 'string' && (equipmentTaskIds as readonly string[]).includes(value);
export const isInspectionTask = (value: unknown): value is InspectionTask => value === 'manual_other' || isFixedTask(value);
