# v0.2.0-rc.1

Recovery and artifact-reproduction candidate for the SCLT manuscript.

- Preserves core solver behavior without numerical changes.
- Restores six later experiment/case-study drivers and selected parameters for three datasets.
- Archives available JSON/CSV evidence with SHA-256 hashes.
- Adds one entry point for integrity checks, tables, plots and isolated cloud runs.
- Preserves the original framework and six latest experimental figures.
- Adds provenance/caveat documentation and tests that do not fit models.
- Excludes raw training tensors, generated reruns, private credentials and submission correspondence.

This is a **prerelease**, not a verified complete rerun. The original server expired; original tensors and environment lockfile are not in this package. Some sweep results survive only as aggregates. See `docs/REPRODUCIBILITY.md` and the validation record for the exact coverage.
