"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const api = require("../review-ui/review-workbench.js");
const meta = {source_package_sha256: "a".repeat(64), weights_sha256: "b".repeat(64), case_ids: ["test-case-1", "test-case-2"], domains: {"test-case-1": "dacl", "test-case-2": "codebrim"}};
function fixture() { return {schema: api.SCHEMA, source_package_sha256: meta.source_package_sha256, weights_sha256: meta.weights_sha256, exported_utc: "2026-10-04T12:00:00.000Z", reviewer: {reviewer_id: "synthetic-observer", name: "합성 테스트", role: "team_observer", expertise: ""}, cases: [{case_id: "test-case-1", tasks: api.TASKS.map(api.blankTask)}]}; }
function observed(doc = fixture()) { Object.assign(doc.cases[0].tasks[0], {judgement: "uncertain", reason: "capture_quality", note: "합성 의견: 사진으로 판단 어려움", reviewed_at: "2026-10-04T11:00:00.000Z"}); return doc; }

test("valid partial opinion roundtrip preserves separate uncertain judgement", () => {
  const doc = observed(), copy = JSON.stringify(doc);
  assert.deepEqual(JSON.parse(JSON.stringify(api.validateFeedback(api.parseStrictJSON(copy), meta))), doc);
  assert.equal(JSON.stringify(doc), copy);
  const counts = api.localCounts(doc, meta);
  assert.equal(counts.reviewed_task_opinions, 1); assert.equal(counts.uncertain_task_opinions, 1);
  assert.equal(counts.label_changes, 0); assert.equal(counts.expert_confirmed_labels, 0);
});
test("expert declaration needs field expertise and evidence including uncertain opinions", () => {
  const doc = observed(); doc.reviewer.role = "domain_expert";
  assert.throws(() => api.validateFeedback(doc, meta)); doc.reviewer.expertise = "합성 분야";
  assert.throws(() => api.validateFeedback(doc, meta)); doc.cases[0].tasks[0].evidence = "합성 근거";
  assert.equal(api.validateFeedback(doc, meta), doc);
});
test("wrong package model unknown IDs and duplicate rows reject atomically", () => {
  for (const mutate of [d => d.source_package_sha256 = "c".repeat(64), d => d.weights_sha256 = "c".repeat(64), d => d.cases[0].case_id = "outside-case", d => d.cases.push(structuredClone(d.cases[0])), d => d.cases[0].tasks[1].task = api.TASKS[0], d => d.cases[0].tasks[0].original_targets = [1]]) {
    const doc = fixture(); mutate(doc); const unchanged = JSON.stringify(doc);
    assert.throws(() => api.validateFeedback(doc, meta)); assert.equal(JSON.stringify(doc), unchanged);
  }
});
test("unreviewed note is not silently treated as reviewed", () => {
  const doc = fixture(); doc.cases[0].tasks[0].note = "아직 판정 선택 전";
  assert.throws(() => api.validateFeedback(doc, meta));
});
test("UTC calendar precision and chronology use the same microsecond boundary", () => {
  for (const date of ["2026-02-30T12:00:00Z", "2026-10-04T12:00:00+09:00", "2026-10-04T12:00:00.0000001Z", "2026-10-04T12:00:60Z"]) {
    const doc = observed(); doc.exported_utc = date; assert.throws(() => api.validateFeedback(doc, meta));
  }
  const doc = observed(); doc.exported_utc = "2026-10-04T12:00:00.000500Z";
  doc.cases[0].tasks[0].reviewed_at = "2026-10-04T12:00:00.000501+00:00";
  assert.throws(() => api.validateFeedback(doc, meta));
  doc.cases[0].tasks[0].reviewed_at = "2026-10-04T12:00:00.000500+00:00";
  assert.equal(api.validateFeedback(doc, meta), doc);
});
test("UTF8 BOM duplicate escaped JSON keys nonfinite and extra trailing values reject", () => {
  for (const source of ['\uFEFF{}', '{"x":1,"\\u0078":2}', '{"x":1e999}', '{} {}', '{"x":NaN}', '[1,]', '{"__proto__":1,"__proto__":2}']) assert.throws(() => api.parseStrictJSON(source));
});
test("identity whitespace control characters and lone surrogates cannot change on import", () => {
  for (const id of [" observer ", "team\nA", "team\rA", "team\u0000A", "team\u007FA", "team\uD800"]) { const doc = fixture(); doc.reviewer.reviewer_id = id; assert.throws(() => api.validateFeedback(doc, meta)); }
  const doc = observed(); doc.cases[0].tasks[0].note = "안전한 Unicode 😀"; assert.equal(api.validateFeedback(doc, meta), doc);
});
test("legacy case level feedback cannot silently become task specific browser opinions", () => {
  const doc = fixture(); doc.schema = "facility_train_review_proposals_v1";
  assert.throws(() => api.validateFeedback(doc, meta));
});
