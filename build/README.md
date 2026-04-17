# Build Infrastructure

## Agent Status
Agents write their status to `build/status.json` every 60 seconds.

## Dashboard
Serve from repo root:
    python -m http.server 8765

Open: http://localhost:8765/build/dashboard/index.html

## Worktrees
| Agent   | Directory          | Branch        |
|---------|--------------------|---------------|
| PHANTOM | ../aria-capture    | feat/capture  |
| MNEMON  | ../aria-memory     | feat/memory   |
| ECHO    | ../aria-output     | feat/output   |
| ORACLE  | ../aria-decide     | feat/decide   |
