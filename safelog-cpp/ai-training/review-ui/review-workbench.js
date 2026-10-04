/* Local review opinions only. This file does not update labels or train models. */
"use strict";
(function () {
  const SCHEMA = "facility_train_review_feedback_v2";
  const TASKS = ["concrete_crack", "concrete_spalling"];
  const REASONS = ["unreviewed", "small_damage", "texture_confusion", "other_damage_confusion", "capture_quality", "annotation_uncertain", "model_error", "correct_comparison", "other"];
  const JUDGEMENTS = ["unreviewed", "present", "absent", "uncertain"];
  const ROLES = ["team_observer", "domain_expert"];
  function fail(message) { throw new Error(message); }
  function object(value, keys, label) {
    if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).some(k => !keys.includes(k)) || keys.some(k => !Object.prototype.hasOwnProperty.call(value, k))) fail(label + " 필드가 잘못되었습니다.");
  }
  function string(value, limit, label, required = false) {
    if (typeof value !== "string" || Array.from(value).length > limit || (required && !value.trim())) fail(label + "을 확인하세요.");
    for (let i = 0; i < value.length; i++) {
      const code = value.charCodeAt(i);
      if (code >= 0xD800 && code <= 0xDBFF) { const next = value.charCodeAt(++i); if (!(next >= 0xDC00 && next <= 0xDFFF)) fail(label + "에 잘못된 Unicode 문자가 있습니다."); }
      else if (code >= 0xDC00 && code <= 0xDFFF) fail(label + "에 잘못된 Unicode 문자가 있습니다.");
    }
  }
  function isoDate(value, label) {
    const match = typeof value === "string" && value.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z|\+00:00)$/);
    if (!match) fail(label + "에 시간대가 포함된 ISO 날짜가 필요합니다.");
    const [year, month, day, hour, minute, second] = match.slice(1, 7).map(Number);
    const days = [31, year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1];
    if (year < 1 || month < 1 || month > 12 || day < 1 || day > days || hour > 23 || minute > 59 || second > 59 || !Number.isFinite(Date.parse(value))) fail(label + " 날짜가 잘못되었습니다.");
    return BigInt(Date.parse(value)) * 1000n + BigInt((match[7] || "").padEnd(6, "0").slice(3));
  }
  // JSON.parse normally discards duplicate keys. Reject them, including escaped keys.
  function parseStrictJSON(source) {
    if (typeof source !== "string" || source.length > 5000000) fail("의견 파일이 너무 큽니다.");
    if (source.startsWith("\uFEFF")) fail("UTF-8 BOM 없이 저장한 JSON이 필요합니다.");
    let index = 0;
    const white = () => { while (/[\t\n\r ]/.test(source[index] || "_") && index < source.length) index++; };
    function quoted() {
      const start = index++;
      while (index < source.length) {
        if (source[index] === "\\") { index += 2; continue; }
        if (source[index++] === '"') return JSON.parse(source.slice(start, index));
      }
      fail("JSON 문자열이 끝나지 않았습니다.");
    }
    function value(depth) {
      if (depth > 50) fail("JSON 중첩이 너무 깊습니다.");
      white();
      if (source[index] === '"') return quoted();
      if (source[index] === "{") {
        index++; white(); const result = Object.create(null), seen = new Set();
        if (source[index] === "}") { index++; return result; }
        while (true) {
          white(); if (source[index] !== '"') fail("JSON 객체 키가 잘못되었습니다.");
          const key = quoted(); if (seen.has(key)) fail("중복 JSON 키: " + key); seen.add(key);
          white(); if (source[index++] !== ":") fail("JSON 콜론이 필요합니다.");
          result[key] = value(depth + 1); white();
          if (source[index] === "}") { index++; return result; }
          if (source[index++] !== ",") fail("JSON 쉼표가 필요합니다.");
        }
      }
      if (source[index] === "[") {
        index++; white(); const result = [];
        if (source[index] === "]") { index++; return result; }
        while (true) { result.push(value(depth + 1)); white(); if (source[index] === "]") { index++; return result; } if (source[index++] !== ",") fail("JSON 배열이 잘못되었습니다."); }
      }
      const token = source.slice(index).match(/^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
      if (!token) fail("JSON 값이 잘못되었습니다.");
      index += token[0].length; const result = JSON.parse(token[0]);
      if (typeof result === "number" && !Number.isFinite(result)) fail("JSON 숫자가 유한하지 않습니다.");
      return result;
    }
    const result = value(0); white(); if (index !== source.length) fail("JSON 뒤에 잘못된 내용이 있습니다."); return result;
  }
  function validateFeedback(doc, metadata) {
    object(doc, ["schema", "source_package_sha256", "weights_sha256", "exported_utc", "reviewer", "cases"], "의견 파일");
    if (doc.schema !== SCHEMA) fail("새 v2 의견 파일만 불러올 수 있습니다. 기존 v1 의견은 CLI에서 별도로 집계하세요.");
    if (doc.source_package_sha256 !== metadata.source_package_sha256 || doc.weights_sha256 !== metadata.weights_sha256) fail("다른 사진 묶음 또는 모델의 의견 파일입니다.");
    const exported = isoDate(doc.exported_utc, "저장 날짜");
    object(doc.reviewer, ["reviewer_id", "name", "role", "expertise"], "검수자");
    string(doc.reviewer.reviewer_id, 200, "검수자 ID", true); string(doc.reviewer.name, 200, "검수자 이름", true); string(doc.reviewer.expertise, 500, "전문 분야");
    if (doc.reviewer.reviewer_id !== doc.reviewer.reviewer_id.trim()) fail("검수자 ID 앞뒤의 공백을 제거하세요.");
    if ([doc.reviewer.reviewer_id, doc.reviewer.name, doc.reviewer.expertise].some(value => /[\x00-\x1F\x7F]/.test(value))) fail("검수자 정보에는 줄바꿈이나 제어 문자를 넣을 수 없습니다.");
    if (!ROLES.includes(doc.reviewer.role)) fail("검수자 구분이 잘못되었습니다.");
    if (!Array.isArray(doc.cases) || !doc.cases.length || doc.cases.length > metadata.case_ids.length) fail("의견 사진 수가 잘못되었습니다.");
    const seen = new Set();
    for (const row of doc.cases) {
      object(row, ["case_id", "tasks"], "사진 의견");
      if (!metadata.case_ids.includes(row.case_id) || seen.has(row.case_id)) fail("미등록 또는 중복 사진 ID입니다."); seen.add(row.case_id);
      if (!Array.isArray(row.tasks) || row.tasks.length !== 2) fail("균열·박락 의견이 각각 하나씩 필요합니다.");
      const tasks = new Set();
      for (const entry of row.tasks) {
        object(entry, ["task", "judgement", "reason", "note", "evidence", "reviewed_at"], "항목 의견");
        if (!TASKS.includes(entry.task) || tasks.has(entry.task)) fail("미등록 또는 중복 손상 항목입니다."); tasks.add(entry.task);
        if (!JUDGEMENTS.includes(entry.judgement) || !REASONS.includes(entry.reason)) fail("판단 또는 사유가 잘못되었습니다.");
        string(entry.note, 4000, "관찰 메모"); string(entry.evidence, 4000, "판단 근거");
        if (entry.judgement === "unreviewed") {
          if (entry.reason !== "unreviewed" || entry.note.trim() || entry.evidence.trim() || entry.reviewed_at !== null) fail("검수 전 항목의 입력을 지우거나 판단을 선택하세요.");
        } else {
          if (entry.reason === "unreviewed" || !entry.note.trim()) fail("검수한 항목에는 사유와 관찰 메모가 필요합니다.");
          if (isoDate(entry.reviewed_at, "의견 날짜") > exported) fail("의견 날짜가 저장 날짜보다 늦습니다.");
          if (doc.reviewer.role === "domain_expert" && (!doc.reviewer.expertise.trim() || !entry.evidence.trim())) fail("전문가 의견에는 전문 분야와 항목별 판단 근거가 필요합니다.");
        }
      }
    }
    return doc;
  }
  function blankTask(task) { return {task, judgement: "unreviewed", reason: "unreviewed", note: "", evidence: "", reviewed_at: null}; }
  function localCounts(doc, metadata) {
    const counts = {total_cases: metadata.case_ids.length, reviewed_cases: 0, reviewed_task_opinions: 0, uncertain_task_opinions: 0, by_reason: {}, by_domain: {}, label_changes: 0, expert_confirmed_labels: 0};
    for (const row of doc.cases) {
      let reviewed = false;
      for (const entry of row.tasks) if (entry.judgement !== "unreviewed") {
        reviewed = true; counts.reviewed_task_opinions++;
        if (entry.judgement === "uncertain") counts.uncertain_task_opinions++;
        counts.by_reason[entry.reason] = (counts.by_reason[entry.reason] || 0) + 1;
        const domain = metadata.domains[row.case_id]; counts.by_domain[domain] = (counts.by_domain[domain] || 0) + 1;
      }
      if (reviewed) counts.reviewed_cases++;
    }
    return counts;
  }
  const api = {SCHEMA, TASKS, REASONS, JUDGEMENTS, ROLES, parseStrictJSON, validateFeedback, blankTask, localCounts};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;
  const metadata = window.reviewMetadata;
  const cards = Array.from(document.querySelectorAll("article[data-case-id]"));
  let dirty = false;
  const status = document.getElementById("status"), feedbackInput = document.getElementById("import");
  const get = id => document.getElementById(id);
  const notify = (message, error = false) => { status.textContent = message; status.className = error ? "error" : "success"; };
  function reviewer() { return {reviewer_id: get("reviewer-id").value.trim(), name: get("reviewer-name").value.trim(), role: get("reviewer-role").value, expertise: get("reviewer-expertise").value.trim()}; }
  function taskState(fieldset) {
    return {task: fieldset.dataset.task, judgement: fieldset.querySelector(".judgement").value, reason: fieldset.querySelector(".reason").value, note: fieldset.querySelector(".note").value, evidence: fieldset.querySelector(".evidence").value, reviewed_at: fieldset.dataset.reviewedAt || null};
  }
  function snapshot() { return {schema: SCHEMA, source_package_sha256: metadata.source_package_sha256, weights_sha256: metadata.weights_sha256, exported_utc: new Date().toISOString(), reviewer: reviewer(), cases: cards.map(a => ({case_id: a.dataset.caseId, tasks: Array.from(a.querySelectorAll("fieldset[data-task]")).map(taskState)}))}; }
  function refresh() {
    const doc = snapshot(), counts = localCounts(doc, metadata);
    get("progress").textContent = `입력된 사진 ${counts.reviewed_cases}/${counts.total_cases} · 항목 의견 ${counts.reviewed_task_opinions} · 판단 어려움 ${counts.uncertain_task_opinions} · 정답 수정 0건`;
    const reasonLabels = {small_damage: "작은 손상", texture_confusion: "질감·그림자·이음매", other_damage_confusion: "다른 손상", capture_quality: "촬영 상태", annotation_uncertain: "정답 기준 질문", model_error: "모델 오류 의심", correct_comparison: "일치 비교", other: "기타"};
    get("reason-counts").textContent = Object.entries(counts.by_reason).map(([k, v]) => `${reasonLabels[k] || k}: ${v}`).join(" / ") || "기록된 원인 의견이 없습니다.";
    for (let i = 0; i < cards.length; i++) {
      const reviewed = doc.cases[i].tasks.some(t => t.judgement !== "unreviewed");
      cards[i].hidden = (get("domain-filter").value !== "all" && cards[i].dataset.domain !== get("domain-filter").value) || (get("pending-only").checked && doc.cases[i].tasks.every(t => t.judgement !== "unreviewed"));
      cards[i].querySelector(".review-state").textContent = reviewed ? "의견 입력 있음" : "검수 전";
    }
    document.body.classList.toggle("show-reference", get("show-reference").checked);
    document.body.classList.toggle("show-model", get("show-model").checked);
    document.body.classList.toggle("expert-mode", get("reviewer-role").value === "domain_expert");
  }
  function applyFeedback(doc) {
    // Validate before touching any input. Partial imports replace the entire current draft.
    validateFeedback(doc, metadata);
    const byId = new Map(doc.cases.map(c => [c.case_id, c]));
    get("reviewer-id").value = doc.reviewer.reviewer_id; get("reviewer-name").value = doc.reviewer.name; get("reviewer-role").value = doc.reviewer.role; get("reviewer-expertise").value = doc.reviewer.expertise;
    for (const card of cards) {
      const rows = new Map((byId.get(card.dataset.caseId)?.tasks || TASKS.map(blankTask)).map(t => [t.task, t]));
      for (const fieldset of card.querySelectorAll("fieldset[data-task]")) {
        const entry = rows.get(fieldset.dataset.task);
        for (const key of ["judgement", "reason", "note", "evidence"]) fieldset.querySelector("." + key).value = entry[key];
        fieldset.dataset.reviewedAt = entry.reviewed_at || "";
      }
    }
    dirty = false; refresh();
  }
  for (const card of cards) for (const fieldset of card.querySelectorAll("fieldset[data-task]")) {
    fieldset.addEventListener("input", () => {
      dirty = true;
      if (fieldset.querySelector(".judgement").value !== "unreviewed") fieldset.dataset.reviewedAt = new Date().toISOString();
      else fieldset.dataset.reviewedAt = "";
      refresh();
    });
    fieldset.querySelector(".reset-task").addEventListener("click", () => {
      for (const key of ["judgement", "reason", "note", "evidence"]) fieldset.querySelector("." + key).value = blankTask(fieldset.dataset.task)[key];
      fieldset.dataset.reviewedAt = ""; dirty = true; refresh(); notify("해당 항목 의견을 지웠습니다. 원본 정답은 그대로입니다.");
    });
  }
  for (const id of ["reviewer-id", "reviewer-name", "reviewer-role", "reviewer-expertise"]) get(id).addEventListener("input", () => { dirty = true; refresh(); });
  for (const id of ["domain-filter", "pending-only", "show-reference", "show-model"]) get(id).addEventListener("change", refresh);
  get("export").addEventListener("click", () => {
    try {
      const doc = validateFeedback(snapshot(), metadata);
      const encoded = JSON.stringify(doc, null, 2);
      get("export-json").value = encoded; get("export-preview").open = true;
      const blob = new Blob([encoded], {type: "application/json;charset=utf-8"});
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = `TRAIN-feedback-${metadata.source_package_sha256.slice(0, 8)}-${doc.exported_utc.replace(/[:.]/g, "-")}.json`;
      link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      notify("의견 JSON 다운로드를 요청했습니다. 파일이 저장되지 않으면 아래 JSON을 복사해 UTF-8 파일로 저장하세요.");
    } catch (error) { notify(error.message, true); }
  });
  get("select-json").addEventListener("click", () => { get("export-json").focus(); get("export-json").select(); });
  feedbackInput.addEventListener("change", async () => {
    const file = feedbackInput.files[0]; if (!file) return;
    try {
      if (file.size > 5000000) fail("5MB 이하의 의견 파일만 불러올 수 있습니다.");
      const source = new TextDecoder("utf-8", {fatal: true, ignoreBOM: true}).decode(await file.arrayBuffer());
      const doc = validateFeedback(parseStrictJSON(source), metadata);
      if (dirty && !window.confirm("저장하지 않은 입력을 불러온 의견으로 교체할까요? 먼저 JSON 저장을 권장합니다.")) return;
      applyFeedback(doc); notify("의견을 불러왔습니다. 이 검수자의 입력을 이어서 작성할 수 있습니다.");
    } catch (error) { notify(error.message, true); }
    finally { feedbackInput.value = ""; }
  });
  window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
  refresh();
})();
