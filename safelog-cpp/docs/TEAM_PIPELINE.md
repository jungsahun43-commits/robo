# Four-person development pipeline

## Common preparation (half day)

1. Build and run `safelog_demo` on one computer.
2. Freeze the shared contract headers.
3. Create branches `feature/capture`, `feature/storage`, `feature/workflow`, and `feature/reporting`.
4. Each member changes only owned paths until the integration day.
5. Every branch must keep its module buildable and add or update a module test.

## A — field capture

Owned paths:

- `src/capture/**`
- later: `apps/mobile-qt/qml/capture/**`

Input: site id, inspector id, location, description, action opinion, local photo path.

Output: persisted `Inspection`, `Finding`, before `Photo`, and creation `ActionLog`.

Next tasks:

- Allow multiple before photos.
- Add draft editing before final submission.
- Implement Qt camera/gallery picker and permission errors.
- Add validation tests for blank fields and missing photos.

## B — storage and platform data

Owned paths:

- `src/storage/**`
- `database/**`

Input/output contract: `IRepository`, `IClock`, `IIdGenerator`, `IPhotoStore`.

Next tasks:

- Implement `SqliteRepository` with Qt SQL.
- Run schema migrations by version.
- Copy content URI images into the app data directory.
- Add seed data and backup/export.
- Verify persistence after app restart.

## C — assignment and corrective action

Owned paths:

- `src/workflow/**`
- later: `apps/mobile-qt/qml/workflow/**`

Input: finding id, acting user id, action note, after photo path.

Output: validated status transition, updated finding, after photo, audit log.

Next tasks:

- Add rejection from `pending_review` back to `in_progress` with a reason.
- Add due-date and overdue queries.
- Build assigned-task and before/after comparison screens.
- Add one negative test for every invalid transition.

## D — report pipeline

Owned paths:

- `src/reporting/**`
- later: `apps/mobile-qt/qml/reporting/**`

Input: inspection id through `ReportService`.

Output: complete `ReportData`, HTML preview, and later a PDF file.

Next tasks:

- Build a `QPdfWriter` renderer and embed local images.
- Add missing-field validation and a preview screen.
- Add Android file sharing adapter.
- Compare the output against the team's chosen sample inspection form.

## Integration day order

1. Merge storage because every feature depends on the repository contract.
2. Merge capture and run a create/reopen scenario.
3. Merge workflow and run all four transitions.
4. Merge reporting and inspect the final PDF on Android.
5. Fix integration defects in the module that owns the behavior; the integrator does not rewrite every module.

## Required demonstration

```text
Inspector creates finding with before photo
Manager assigns it
Assignee starts work and submits note + after photo
Inspector verifies it
App generates and shares a report containing the full audit trail
```
