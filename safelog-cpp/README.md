# SafeLog C++ prototype

SafeLog AI is a contract-first C++20 project for AI-assisted workplace safety inspection. A multimodal analyzer suggests hazards and corrective action from a before photo, compares before/after photos, and drafts a report summary. A human reviews every AI result and makes the final decision.

## Latest facility AI research (2026-10-11)

Spalling hard-background pixel mining completed six actual epochs (19.89 minutes). Matched public validation maximum FNR/FPR: 22.0207% control versus 22.5309% candidate. Small-defect false negatives: 70 versus 70. The candidate was not adopted; each rate below5% remains unmet.

[Study results](ai-training/reports/FACILITY_SPALLING_OHEM_STUDY_RESULTS_KO.md) · [Per-domain rate tradeoffs](ai-training/reports/FACILITY_SPALLING_OHEM_BREAKDOWN_KO.md). The application default model remains unchanged; these are repeated public-source validation results.

## Previous ConvNeXt facility study (2026-10-10)

A fully trainable ConvNeXt facility model completed six actual epochs. Matched public validation worst crack/spalling FNR/FPR: 22.0207% control versus 22.2798% candidate. Each rate below5% remains unmet. This is repeated source validation, not independent workplace accuracy.

[Study results](ai-training/reports/FACILITY_CONVNEXT_STUDY_RESULTS_KO.md) · [Research checkpoint handoff and local inference timing](ai-training/reports/FACILITY_CONVNEXT_HANDOFF_KO.md). The application's default model remains unchanged.

## Current vertical slice

The included CLI demo executes the complete scenario:

1. Create an inspection.
2. Register a finding with a before photo.
3. Run mock AI hazard analysis and record the inspector's review.
4. Assign a corrective-action owner.
5. Start and submit corrective action with an after photo.
6. Run mock AI before/after comparison and record human review.
7. Let the original inspector verify it.
8. Generate an AI summary and an HTML report containing AI and human audit trails.

The Qt shell is optional because Qt is a large separate installation. Enable it after installing Qt 6.5 or newer with Quick and Quick Controls 2.

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/apps/demo/safelog_demo
```

Qt desktop/Android shell:

```bash
cmake -S . -B build-qt -DSAFELOG_BUILD_QT=ON -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x/compiler
cmake --build build-qt
```

## Team ownership

| Member | Library | Primary result |
|---|---|---|
| A | capture UI | photo input and human review of AI hazard suggestions |
| B | storage | SQLite, photos, AI analysis and review persistence |
| C | `src/ai` | real multimodal model adapter and evaluation |
| D | workflow/report UI | corrective action, AI comparison, final report |

Use `docs/AI_FIRST_TEAM_PROMPTS_KO.md` for the current assignment. `MockAiSafetyAnalyzer` keeps all teams unblocked before a real model server is available.

## Project boundaries

This prototype assists record creation and corrective-action tracking. It does not automatically determine whether a workplace is legally compliant and should not be presented as a certified statutory form without review of the target industry and actual form.
