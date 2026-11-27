import type { Analysis } from '@/contracts/types';

import { inspectionCategories, inspectionCategoryIds } from '@/features/ai/inspection-categories';

const groupDefinitions = [
  { label: '시설·컨베이어 (3개)', ids: ['facility_crack', 'belt_damage', 'conveyor_foreign'] },
  { label: '전기·배선 (2개)', ids: ['cable_exterior', 'loco_splicing_connectors'] },
  { label: '금속·기계 부품 (10개)', ids: ['ad_screw', 'ad_metal_nut', 'mpdd_bracket_black', 'mpdd_bracket_brown',
    'mpdd_bracket_white', 'mpdd_connector', 'mpdd_metal_plate', 'mpdd_tubes', 'ksdd2_surface', 'ball_screw_pitting'] },
  { label: '회로기판 (5개)', ids: ['pcb1', 'pcb2', 'pcb3', 'pcb4', 'robustad_pcb'] },
] as const;

export const checklistGroups = groupDefinitions.map(({ label, ids }) => ({
  label,
  items: ids.map(id => [id, id === 'facility_crack' ? '시설 균열 등 표면 손상' : inspectionCategories[id].label,
    inspectionCategories[id].check] as const),
}));

export const inspectionColumns = ['구분', '점검 항목', '해당 여부', '적정', '부적정', '조치사항', '조치 완료'] as const;

export type ChecklistRow = { applicable?: boolean; result?: 'adequate' | 'inadequate'; note?: string; completed?: boolean; suggestedByAi?: boolean };
export type Checklist = Partial<Record<string, ChecklistRow>>;

const categoryItems = new Map<string, string>(inspectionCategoryIds.flatMap(id => [
  [id, id], [inspectionCategories[id].label, id],
]));

export function latestInspectionCategory(analyses: readonly Analysis[]) {
  const analysis = analyses.filter(item => item.type === 'BeforeHazard').at(-1);
  if (!analysis || analysis.decision === 'Rejected') return;
  const task = analysis.source === 'fixed-cpu' ? analysis.inspectionTask : undefined;
  const category = task === 'facility_surface' ? 'facility_crack' : task === 'ad_cable' ? 'cable_exterior' : task;
  const id = categoryItems.get(category ?? analysis.category?.trim() ?? '') as keyof typeof inspectionCategories | undefined;
  if (id) return { id, label: inspectionCategories[id].label, mock: analysis.model === 'safelog-mobile-mock' };
}

export function checklistWithAi(analyses: readonly Analysis[], manual: Checklist): Checklist {
  const id = latestInspectionCategory(analyses)?.id;
  if (!id) return manual;
  const row = manual[id];
  if (row && Object.hasOwn(row, 'applicable')) return manual;
  return { ...manual, [id]: { ...row, applicable: true, suggestedByAi: true } };
}

export function escapeHtml(text: string) {
  return text.replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char] ?? char);
}

const mark = (checked?: boolean) => checked ? '☑' : '□';

export function preliminarySheetHtml(
  info: readonly (readonly [string, string])[],
  checklist: Checklist,
) {
  const [date, ...metadata] = info;
  const work = metadata.pop();
  const cells = ([label, value]: readonly [string, string], span = '') =>
    '<th scope="row">' + escapeHtml(label) + '</th><td' + span + '>' + escapeHtml(value) + '</td>';
  const infoRows = metadata.map((entry, index) => {
    if (index % 2 !== 0) return '';
    const next = metadata[index + 1];
    return '<tr>' + (next ? cells(entry) + cells(next) : cells(entry, ' colspan="3"')) + '</tr>';
  }).join('') + (work ? '<tr>' + cells(work, ' colspan="3"') + '</tr>' : '');
  const groups = checklistGroups.map(({ label, items }) =>
    '<tbody><tr class="group"><th colspan="7">' + escapeHtml(label) + '</th></tr>' +
    items.map(([id, category, description]) => {
      const row = checklist[id];
      const columns = [category, description, row?.applicable === false ? '해당 없음' : mark(row?.applicable) + (row?.suggestedByAi ? ' (AI 제안)' : ''),
        mark(row?.result === 'adequate'), mark(row?.result === 'inadequate'), row?.note ?? '', mark(row?.completed)];
      return '<tr>' + columns.map((value, index) =>
        '<td' + ([2, 3, 4, 6].includes(index) ? ' class="check"' : '') + '>' + escapeHtml(value) + '</td>').join('') + '</tr>';
    }).join('') + '</tbody>').join('');
  return '<section class="sheet"><header><div>설비·시설 사진 외관 점검표</div>' +
    '<h1>작 업 장 순 회 점 검 표</h1><strong>예비 안전점검 보고서</strong></header>' +
    (date ? '<div class="sheet-date">' + escapeHtml(date.join(' · ')) + '</div>' : '') +
    '<table class="sheet-info"><tbody>' + infoRows + '</tbody></table>' +
    '<table class="sheet-items"><colgroup><col style="width:12%"><col style="width:39%"><col style="width:8%">' +
    '<col style="width:7%"><col style="width:7%"><col style="width:19%"><col style="width:8%"></colgroup>' +
    '<thead><tr>' + inspectionColumns.map(label => '<th>' + label + '</th>').join('') + '</tr></thead>' +
    groups + '</table></section>';
}

export const preliminarySheetCss = '.sheet{font-size:10px;color:#111}.sheet header{text-align:center;margin-bottom:10px}' +
  '.sheet h1{font-size:19px;margin:3px 0 6px}.sheet table{border-collapse:collapse;width:100%;table-layout:fixed;margin:0 0 10px}' +
  '.sheet th,.sheet td{border:1px solid #222;padding:4px;overflow-wrap:anywhere;vertical-align:middle;white-space:pre-wrap}' +
  '.sheet-date{text-align:right;margin:0 0 5px}.sheet-info th{width:15%;background:#e1e7f3}.sheet .group th{background:#e1e7f3;text-align:left}' +
  '.sheet .check{text-align:center}.sheet tr{break-inside:avoid}.sheet thead{display:table-header-group}' +
  '@media print{.sheet th{-webkit-print-color-adjust:exact;print-color-adjust:exact}}';
