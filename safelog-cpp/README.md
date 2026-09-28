# SafeLog C++ prototype

SafeLog AI is a contract-first C++20 project for AI-assisted workplace safety inspection. A multimodal analyzer suggests hazards and corrective action from a before photo, compares before/after photos, and drafts a report summary. A human reviews every AI result and makes the final decision.

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
