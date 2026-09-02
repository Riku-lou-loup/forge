# Data and documents

The development sample is the SKAB recording `raw/skab/valve1_1.csv`. Its exact
source revision and SHA-256 checksum are recorded in `sample-manifest.json`.
The CSV is ignored by Git. Recreate it using the pinned source:

```powershell
.\.venv\Scripts\python.exe -m forge.data.sample --download
```

This file is for development exploration and remains in the training partition.
The [full inventory](skab-inventory.json) pins 35 files, and the
[split manifest](skab-splits.json) assigns complete overlap groups to training,
validation, and test partitions. See the [split protocol](../docs/data-split.md)
for allocation, overlap findings, reproduction commands, and limitations.

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
and two annotation columns. The flow column is named
`Volume Flow RateRMS`. Use this CSV header when loading the data, because the
top-level source README uses a shorter name. Times span 20 minutes. There are
1,088 one-second intervals and 56 two-second intervals. All recorded cells contain
values, and the analysis preserves the gaps between timestamps.

Choose technical documents that apply to the supported asset. Identify generic
references and project-authored demonstration procedures clearly, and check a
manufacturer's manual for relevance before pairing it with testbed measurements.

SKAB and any later live setup require separate validation. An unusual signal
does not establish a root cause or remaining useful life.
