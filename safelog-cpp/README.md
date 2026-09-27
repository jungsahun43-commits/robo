# SafeLog C++ prototype

SafeLog is a contract-first C++20 project for recording workplace safety findings, tracking corrective action, and generating an inspection report. It is structured for four people to build independent modules and connect them at the end.

## Current vertical slice

The included CLI demo executes the complete scenario:

1. Create an inspection.
2. Register a finding with a before photo.
3. Assign a corrective-action owner.
4. Start and submit corrective action with an after photo.
5. Let the original inspector verify it.
6. Generate an HTML report containing the audit trail.

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
| A | `src/capture` | inspection and before-photo registration |
| B | `src/storage`, `database` | SQLite, files, ids, persistence |
| C | `src/workflow` | assignment and validated state transitions |
| D | `src/reporting` | report validation, preview, PDF/export |

See `docs/TEAM_PIPELINE.md` before creating branches. The three headers under `include/safelog/contracts` are the shared API and should be frozen first.

## Project boundaries

This prototype assists record creation and corrective-action tracking. It does not automatically determine whether a workplace is legally compliant and should not be presented as a certified statutory form without review of the target industry and actual form.
