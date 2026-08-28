# Data and documents

The development sample is one SKAB recording: `raw/skab/valve1_1.csv`. Its exact
source revision and SHA-256 checksum are recorded in `sample-manifest.json`.
The CSV is ignored by Git. Recreate it using the pinned source:

```powershell
.\.venv\Scripts\python.exe -m forge.data.sample --download
```

This file is for development exploration and must not become part of the final
held-out test set. The remaining experiment inventory and full split are still
to be defined.

The source is [SKAB](https://github.com/waico/SKAB).
Read its [experiment catalog](https://github.com/waico/SKAB/blob/master/data/README.md)
and the license information for the exact artifacts before downloading them.

- `raw/`: unchanged source recordings, excluded from Git.
- `processed/`: derived data, excluded from Git.
- `documents/`: permitted technical references, excluded from Git.

Keep a tracked manifest in this folder with source URL, revision, download date,
checksums, experiment IDs, units, label meaning, and attribution. Do not commit
large data files. Hold out whole experiments and inspect related runs for leakage.
Never feed anomaly or change-point labels into the detector.

The inspected sample contains 1,145 rows, eight sensors, a timestamp column,
and two annotation columns. The flow column is actually named
`Volume Flow RateRMS`; preserve the CSV header instead of assuming the shorter
name used in the top-level source README. Times span 20 minutes. There are
1,088 one-second intervals and 56 two-second intervals; no missing cells occur
in the recorded rows. These sampling gaps are preserved, not filled.

Use documents appropriate to the supported asset. Clearly identify generic
references and project-authored demonstration procedures. A public testbed's
measurements do not justify attaching an unrelated manufacturer's manual.

SKAB and any later live setup require separate validation. An unusual signal
does not establish a root cause or remaining useful life.
