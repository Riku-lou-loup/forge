# PyTorch and BM25 increment

This demo adds two working components to FORGE: a small temporal neural network
trained with PyTorch, and BM25 retrieval over the existing analytical notes.
They can run together through the investigation graph. The active model remains
the gradient-boosting detector, and default investigations still use TF-IDF.

Open [notebook 06](../notebooks/06_pytorch_bm25_demo.ipynb) for the training loss,
validation confusion matrix, an inline recording chart, retrieval comparison,
and a cited investigation using the research checkpoint.

## Reproduce locally

Use Python 3.13 and the project virtual environment. After the base setup, install
the optional CPU training dependencies from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.\.venv\Scripts\python.exe -m forge.data.partitions --split train --download --audit
.\.venv\Scripts\python.exe -m forge.data.partitions --split validation --download --audit
.\.venv\Scripts\python.exe -m forge.rag.evaluate --compare
.\.venv\Scripts\python.exe -m forge.ml.torch_training
```

The optional dependency snapshot was tested on Windows with CPU PyTorch 2.14.1.
The base app does not require it. For another supported platform, the package
extra `.[torch]` declares the dependency, but that platform has not been verified
here. See the [official installation instructions](https://pytorch.org/get-started/locally/)
when selecting a different build.

Training prints its run directory under `models/pytorch-*`. To use the newest
local research run with BM25:

```powershell
$run = Get-ChildItem .\models -Directory -Filter 'pytorch-*' |
    Sort-Object Name | Select-Object -Last 1
.\.venv\Scripts\python.exe -m forge investigate --recording valve1/1 --retriever bm25 --torch-artifact $run.FullName --export
```

This writes an unreviewed draft under `reports/incidents`. Pass a specific run
directory to compare an older checkpoint. Relative artifact paths are resolved
from the repository root. Omitting `--torch-artifact` uses the existing active
model. Omitting `--retriever` uses TF-IDF. Neither option changes a model pointer.
Raw recordings, checkpoints and exported reports remain outside Git.

## What the increment shows

The fixed eight-epoch PyTorch run reached 92.7% precision at 31.4% recall on the
existing validation allocation: 1,100 true positives, 86 false positives, 2,407
missed annotated anomalies and 6,380 true negatives. It produced seven false
alert onsets. There were 289 unavailable readings, including 31 annotated
anomalies, while startup windows were being accumulated. Unavailable anomalies
remain in the recall denominator.

The threshold was selected on this same validation data to prioritize precision
subject to a 30% recall floor. The low recall and two groups with no detected
anomalous readings limit the result. This is evidence that the training and
inference pipeline works. It does not establish an improvement over the grouped
audit or performance on new equipment. The architecture and training budget
were fixed before this run, and the original test recordings were not opened.

BM25 and TF-IDF both reached recall at 3 and reciprocal rank of 1.0 for the seven
answerable regression queries and abstained on both unrelated queries. BM25
returned 15 verified passages in total, compared with seven for TF-IDF. This
fixture was authored with the seven-passage corpus and was already perfect on
its expected results. It does not show that BM25 is more useful, and citation
verification alone does not establish relevance.

The combined graph scores a recording, retrieves notes for the observed sensor
changes, checks the citations and produces analytical checks for review. The
network predicts source anomaly annotations. It does not identify confirmed
physical failures. Retrieval uses project-authored notes, and the graph does
not call an LLM.

[Training details and limitations](pytorch-demo.md) |
[BM25 scoring and per-query comparison](bm25-demo.md)
