# Following one investigation through the code

The application starts with a recording and a frozen local detector. A report is
the output of several small modules; the notebook is preserved as the exploratory
record, while reusable behavior lives in the package.

## From source files to scores

`data/partitions.py` checks the inventory and fixed split before opening pinned
CSV files. `data/datasets.py::normal_training` selects normal training rows and
deduplicates them. `sensor_matrix` selects exactly eight sensor columns, so labels
and identifiers cannot slip into fitting or scoring.

`ml/detectors.py::Detector.fit` computes a median and a robust scale for each
sensor. The statistical score is the largest absolute deviation across sensors:

```text
deviation = (reading - training median) / training scale
statistical score = max(abs(deviation))
```

The scale is 1.4826 × MAD. Pressure has zero MAD even though it varies, so its
scale falls back to IQR / 1.349. If both spreads are zero, fitting stops instead
of dividing by a tiny arbitrary constant. The Isolation Forest alternative
fits trees on the same raw eight-column normal data. Its negative score_samples
output increases for more unusual observations.

`ml/training.py::train` fits the three fixed candidates, evaluates their validation
thresholds, and saves the winner. The threshold is selected jointly with a fixed
three-observation persistence rule. `ml/metrics.py::causal_alerts` only turns an
alert on at the third qualifying observation; future measurements never change
earlier alert values. `evaluate` measures both point classification and event
onsets, because a high point recall can hide warnings that started too early.

## From an alert to a report

`agents/service.py` loads a trusted artifact and a permitted development recording.
`agents/workflow.py::investigate` removes annotation columns before passing data
to any role. It calculates scores and chooses the persistent episode containing
the largest alerted score. The three largest reference deviations provide sensor
names for retrieval. They are not a fault explanation.

`rag/retrieval.py` indexes seven original passages using word and bigram TF-IDF.
The graph asks for evidence, narrows the query once if necessary, and stops if
nothing supports a check. Drafting copies checks from passages; verification
compares their IDs, versions, hashes and text. No free-form language model fills
in missing facts.

`reports/incident.py` holds the schema, review transition, and exports. Every
report starts unreviewed. The Streamlit controls in `ui/investigation.py` require
a name and acknowledgment before recording review. A draft can still be exported,
but its draft status is retained in the filename and content.

## What to inspect first

1. Compare the pressure notebook calculations with `Detector.fit`.
2. Read the validation table in `docs/evaluation.md`: why can better point F1
   coexist with many more false alert onsets?
3. Open the generated report and compare each check with its cited passage.
4. Read `tests/test_modeling.py::test_alert_before_event_does_not_count_as_detection`
   and the missing-evidence and budget tests in `tests/test_investigation.py`.

The next modeling decision is an explicit alert-burden objective and a fresh
evaluation plan, before a PyTorch sequence model or more agents are introduced.
