import assert from 'node:assert/strict';
import test from 'node:test';
import { ROUTABLE_TASKS } from '../scripts/photo-router.mjs';
import { fixedTaskIds } from '../src/contracts/inspection.ts';
import { checklistGroups, checklistWithAi, latestInspectionCategory, preliminarySheetHtml } from '../src/features/reporting/checklist.ts';

test('routes every fixed inspection task and displays it in the preliminary report', () => {
  assert.equal(fixedTaskIds.length, 20);
  assert.deepEqual([...ROUTABLE_TASKS].sort(), [...fixedTaskIds].sort());
  assert.deepEqual(checklistGroups.map(group => [group.label, group.items.length]), [
    ['시설·컨베이어 (3개)', 3], ['전기·배선 (2개)', 2],
    ['금속·기계 부품 (10개)', 10], ['회로기판 (5개)', 5],
  ]);
  const rows = checklistGroups.flatMap(group => group.items);
  assert.equal(rows.length, 20);
  const selected = new Set();
  for (const task of fixedTaskIds) {
    const analysis = { type: 'BeforeHazard', source: 'fixed-cpu', inspectionTask: task, decision: 'Pending' };
    const category = latestInspectionCategory([analysis]);
    assert.ok(category, task);
    selected.add(category.id);
    const checklist = checklistWithAi([analysis], {});
    assert.deepEqual(checklist[category.id], { applicable: true, suggestedByAi: true });
    const html = preliminarySheetHtml([['점검일자', '2026-11-27'], ['점검 대상 (Part)', category.label]], checklist);
    assert.ok(html.includes(category.label), task);
    assert.ok(html.includes('☑ (AI 제안)'), task);
  }
  assert.equal(selected.size, 20);
});
