# Hospital Readmission Prediction

Predict whether a diabetic patient will be readmitted to the hospital within 30 days of
discharge. ML Zoomcamp 2026 midterm project.

## Problem description

<!-- TODO (write in your own words):
- Why 30-day readmissions matter (patient health, hospital costs, penalties).
- Who uses the model and when: e.g. the care team at discharge time gets a risk score.
- What they do with it: plan a follow-up call or visit for high-risk patients.
- How we measure success: ROC AUC, because only ~11% of encounters are readmitted <30 days.
-->

## Dataset

[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(UCI Machine Learning Repository, CC BY 4.0).

- 101,766 hospital encounters of 71,518 patients, 50 columns.
- Target: `readmitted == "<30"` (11.2% of encounters).

The archive is committed as [`data/data.zip`](data/data.zip). Extract it with:

```bash
uv run python data/download_data.py
```

The script verifies the archive's SHA-256 checksum and downloads it from UCI if it is missing.

## Project structure

| File | Content |
|---|---|
| `notebook.ipynb` | Data cleaning, EDA, feature importance, model selection and tuning |
| `train.py` | Trains the final model and saves it |
| `predict.py` | Loads the model and serves it with FastAPI |
| `Dockerfile` | Container for the prediction service |
| `deploy/` | Kubernetes manifests (kind) |
| `pyproject.toml`, `uv.lock` | Dependencies |

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python version, the virtual
environment and the dependencies.

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync                                  # creates .venv/ with the locked dependencies
uv run python data/download_data.py      # extracts the dataset
uv run jupyter notebook notebook.ipynb   # opens the notebook
```

To activate the environment manually instead of using `uv run`:

```bash
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows
```

## Results

<!-- TODO: models compared, tuned parameters, validation AUC, final model choice. -->

## Training

<!-- TODO (day 3): uv run python train.py -->

## Prediction service

<!-- TODO (day 4): run predict.py locally and example request -->

## Docker

<!-- TODO (day 4): docker build / docker run / example request -->

## Deployment

<!-- TODO (day 5): kind cluster, manifests, port-forward, screenshot of a test request -->
