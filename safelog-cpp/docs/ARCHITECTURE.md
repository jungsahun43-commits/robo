# SafeLog architecture

## One-way dependency rule

```text
Qt/QML or CLI app
        |
        v
capture + workflow + ai + reporting     (independent feature libraries)
        |          |          |
        +----------+----------+
                   v
          contracts/types + ports
                   ^
                   |
        storage adapters (memory / SQLite)
```

- Feature libraries never include another feature library.
- UI code calls feature services and never writes the database directly.
- Storage implements `IRepository` and may be replaced without changing feature code.
- Only `apps/*` constructs concrete objects and connects modules.
- AI providers implement `IAiSafetyAnalyzer`; feature code does not depend on a model vendor.
- Every AI output is stored with model/prompt metadata and a human review decision.

## Stable shared contract

The team freezes these files after the first meeting:

- `include/safelog/contracts/types.hpp`
- `include/safelog/contracts/ports.hpp`
- `include/safelog/contracts/errors.hpp`

Contract changes require all four members to review the same pull request. Additive changes are preferred. Never rename a status string after data has been created.

## State machine

```text
open --assign/begin--> in_progress --submit action + after photo-->
pending_review --inspection owner verifies--> verified
```

An assignment keeps a finding `open`; work starts only when the assigned user accepts it. This separates managerial assignment from physical work.

## Real mobile adapters

The core uses standard C++20. The Android app uses Qt 6:

- Camera/gallery: Qt Multimedia or Android content URI adapter implementing `IPhotoStore`.
- Persistence: Qt SQL SQLite adapter implementing `IRepository` using `database/schema.sql`.
- PDF: `QTextDocument` + `QPdfWriter`, taking `ReportData` from the reporting library.
- Sharing: a small Android JNI adapter that sends an `ACTION_SEND` intent.

Those platform adapters remain outside the domain modules so desktop tests do not require an emulator.
