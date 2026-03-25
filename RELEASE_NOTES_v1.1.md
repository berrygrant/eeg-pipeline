# v1.1

## Summary
v1.1 makes the manual-rejection import path usable for EEGLAB epoched `.set` files that retain rejected trials as flags instead of physically deleting them.

## Highlights
- Manual-rejection imports now honor EEGLAB trial rejection flags.
- Rejection handling is configurable by mode (`manual`, `global`, `manual_or_global`, `all`, `none`).
- Import summaries now report total trials, flagged trials, and retained Standard/Deviant counts.

## Changes
- Added rejection-flag parsing for EEGLAB `reject` metadata during manual `.set` import.
- Excluded flagged epochs before condition recoding and FIF export.
- Added richer import summary metadata for auditing manual-rejection workflows.
- Added regression tests covering reject-mask combination and length validation.

## Validation
- Smoke-tested on EEGLAB epoched input with `rejmanual` flags present.
- Full rerun confirmed corrected retained-epoch counts and metrics generation on a 45-subject post-rejection dataset.

## Known Issues
- GitHub release creation still requires a release object on GitHub for Zenodo to archive the new version.
