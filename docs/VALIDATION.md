# Release Validation

Date: 16 November 2026. Platform: Windows. This is artifact/code-package validation, **not** a fresh cloud experiment run.

| Check | Result |
|---|---|
| Archive checksums and JSON readability | 85 files passed |
| Unit tests without model fitting | 8 passed |
| Python syntax, including UTF-8 BOM sources | Passed |
| Time-series trace arithmetic | 4 methods x 4,224 points; MAE matches saved metadata |
| Experimental figure rebuild | 6 of 6 succeeded |
| Saved vs rebuilt PDF raster comparison | All 6 pixel-identical at a 1,400-pixel longest side |
| Visual inspection | All 6 experimental figures inspected |
| Original framework | SHA-256 unchanged |
| Table 2 | Saved per-seed aggregates match submitted displayed precision |
| Table 3 | Saved aggregate CSV matches submitted displayed precision |
| Table 4 | Saved per-seed aggregates match submitted displayed precision |
| Core source `src/` | No changes from upstream `307b6ba` |
| Credential-pattern scan | No private-key/token/password/server-address candidates in scanned release text |
| Cloud launch without confirmation | Refused before model execution |
| Missing-data preflight | Correctly reports not ready |
| Exported source archive | Extracted into an independent directory; all 8 tests and 6 figure rebuilds passed |
| Full experiment rerun | **Not performed**; original cloud server expired |

The plotting verification environment used Python 3.12.14, NumPy 2.5.5, Matplotlib 3.11.2, pypdf 6.21.4, Pillow 13.0.0 and locally installed Times New Roman. These are observed validation versions, **not** recovered historical experiment dependencies. The compatibility requirement files do not claim to be a fully tested lockfile across operating systems.

Text archives are normalized to LF solely for consistent Git transport; numeric values are unchanged. Figure pixel comparisons are environment-specific. Table precision matching verifies correspondence with saved evidence, not the scientific validity of the original experiment protocols. No model fitting, cloud runtime benchmark or scientific re-estimation was executed during this release.
