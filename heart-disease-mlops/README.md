# Heart Disease Prediction MLOps

This project implements an end-to-end educational MLOps workflow for the
processed Cleveland subset of the UCI Heart Disease dataset: data acquisition
and validation, leakage-safe preprocessing, model comparison and MLflow
tracking, fitted model packaging, automated tests and GitHub Actions
configuration, Docker containerization, a FastAPI prediction service, and
Prometheus-compatible operational monitoring. It is not a clinical tool and no
public-cloud deployment is configured. A local Kubernetes deployment is
provided under `k8s/`.

The final verification report, including actual experiment metrics, a
deliverables checklist, test results, and remaining submission tasks, is
[`reports/final_assignment_report.md`](reports/final_assignment_report.md).

### Implemented architecture

```mermaid
flowchart LR
    UCI[UCI Cleveland processed dataset] --> ACQ[Acquire, validate, clean]
    ACQ --> SPLIT[Stratified train/test split]
    SPLIT --> PREP[Fit preprocessing in CV/training folds]
    PREP --> COMP[Compare Logistic Regression and Random Forest]
    COMP --> MLF[Local MLflow: SQLite + mlartifacts]
    COMP --> EVAL[Held-out test evaluation]
    EVAL --> PKG[Packaged fitted pipeline + metadata]
    PKG --> API[FastAPI /predict]
    API --> HEALTH[/health]
    API --> METRICS[/metrics Prometheus exposition]
    API --> LOGS[Privacy-safe JSON request logs]
    K8S[Docker Desktop Kubernetes] --> API
    PROM[Prometheus scraper] --> METRICS
    PROM --> GRAF[Grafana dashboard]
    TESTS[pytest + Ruff] --> CI[GitHub Actions workflow configured]
    DOCKER[Docker image] --> API
```

This diagram represents local project files and tested local behavior.
GitHub Actions is configured in the workspace, but a successful remote
workflow run has not been verified.

## Project folders

- `data/raw/` — original datasets; do not commit private or licensed data.
- `data/interim/` — temporary data created while preparing the dataset.
- `data/processed/` — cleaned data prepared for later model work.
- `notebooks/` — exploratory analysis and experiments.
- `src/heart_disease/` — reusable Python code for the project.
- `tests/` — automated checks for project code.
- `artifacts/` — generated model files and other machine-learning outputs.
- `reports/` — written summaries and evaluation reports.
- `figures/` — charts and other visual outputs.
- `k8s/` — Kubernetes API Deployment/LoadBalancer Service and Prometheus plus
  Grafana manifests for Docker Desktop Kubernetes.
- `monitoring/` — files and notes for future model monitoring.
- `screenshots/` — screenshots documenting the project.
- `videos/` — project demonstration videos.
- `.github/workflows/` — GitHub Actions workflow definitions.

## Starter files

- `README.md` — explains the project and its folder layout.
- `requirements.txt` — lists the Python packages currently needed by the project.
- `.gitignore` — keeps local environments, secrets, caches, and generated data
  out of Git.
- `src/heart_disease/__init__.py` — marks the source folder as a Python package.
- `tests/__init__.py` — marks the tests folder as a Python package.
- `.github/workflows/mlops-ci.yml` — GitHub Actions workflow configured to run
  Ruff and pytest on pushes and pull requests.

Empty folders are not recorded by Git until they contain a file. The `.gitignore`
rules intentionally keep raw, intermediate, processed data, and generated model
artifacts out of Git; add an approved, non-sensitive placeholder if you later
need Git to preserve one of those empty folders.

## Data acquisition

Step 2 uses the Cleveland processed file from the UCI Heart Disease dataset.
See `data/raw/README.md` for the source, dataset version, column meanings, and
instructions for running the acquisition script. The raw data file is ignored
by Git so it remains local; the dataset documentation is tracked.

## Data cleaning and exploratory analysis

The acquisition script saves the original UCI Cleveland source as
`data/raw/processed.cleveland.data`. `src/heart_disease/eda.py` summarizes the
raw data and saves histograms, a correlation heatmap, target counts, and
continuous-feature boxplots under `reports/figures/`. Continuous features are
used for the correlation and boxplots; coded categories are not treated as
continuous measurements.

`src/heart_disease/data_preprocessing.py` checks documented category and target
codes, writes a separate typed copy to
`data/processed/heart_disease_cleaned.csv`, and leaves the raw file unchanged.
It preserves missing values instead of imputing from the complete dataset;
imputation should be learned from the training split in a later modeling step.
The `num` target keeps its UCI encoding: `0` means no diagnosed disease and
`1`–`4` mean disease is present.

### Verified Step 3 data-quality findings

For the downloaded Cleveland file, the raw dataset has 303 rows and 14 columns.
It has four missing values in `ca`, two in `thal`, no exact duplicate rows, and
target values `0`–`4` (164, 55, 36, 35, and 13 rows, respectively). The
documented category codes observed in the raw file are within their documented
sets. The scripts retain the six missing values and do not collapse the target
to binary or remove observations.

Run these from the project root after installing `requirements.txt`:

```powershell
python src\heart_disease\eda.py
python src\heart_disease\data_preprocessing.py
python src\heart_disease\eda.py --input data\processed\heart_disease_cleaned.csv --format processed
```

## Feature engineering and baseline models

The processed data contains 303 rows and target `num`, with documented values
0–4. For the assignment's presence/absence goal, training explicitly maps
`num == 0` to absence (`0`) and `num > 0` to presence (`1`); it keeps all 13
input features. The observed target counts are 164 absence and 139 presence.

Run the following commands from the project root:

```powershell
python src\heart_disease\feature_engineering.py
python src\heart_disease\train.py
python src\heart_disease\evaluate.py
```

The split is stratified (80% training, 20% test) with random seed 42. Numerical
features are median-imputed and scaled; documented category-code features are
most-frequent-imputed and one-hot encoded. These transformations are within each
scikit-learn pipeline, so fitting happens on training rows (and separately
inside each cross-validation fold), never on the full dataset before splitting.
The feature names and target mapping are checked against the Step 2/3 dataset.

Both Logistic Regression and Random Forest use the same training split. The
evaluation script compares their mean ROC-AUC over five stratified folds on
training data and selects the model with the higher mean. It then reports
accuracy, precision, recall, ROC-AUC, a confusion matrix, and a classification
report on the held-out test set. Test scores are for final reporting and are not
used to tune or select the model.

The trained candidates are saved to `artifacts/candidate_models.joblib`;
evaluation saves the selected pipeline, including fitted preprocessing, to
`artifacts/selected_model.joblib`. Cross-validation and test metrics are saved
to `reports/model_evaluation.json`, and test confusion matrices are saved in
`reports/figures/`. Generated model files are ignored by Git. These baseline
metrics are educational estimates, not evidence of clinical performance.

### Step 4 baseline results

The fixed split contains 242 training rows and 61 test rows. The model is
selected by mean five-fold training cross-validation ROC-AUC, not by test score:

| Model | CV ROC-AUC mean +/- standard deviation | Test accuracy | Test precision (presence) | Test recall (presence) | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.902 +/- 0.016 | 0.885 | 0.839 | 0.929 | 0.966 |
| Random Forest | 0.894 +/- 0.037 | 0.869 | 0.812 | 0.929 | 0.944 |

Logistic Regression was selected because it had the higher mean training-fold
ROC-AUC. For its test confusion matrix, 28 absence records and 26 presence
records were correctly classified; 5 absence records were predicted as
presence, and 2 presence records were predicted as absence. The test scores
come from one held-out split of a small dataset and should not be interpreted as
clinical evidence or as a guarantee of future performance.

## Step 5: experiment tracking with MLflow

Experiment tracking records each model attempt in a consistent place: its
settings (parameters), measured results (metrics), and useful files or models
(artifacts). MLflow makes it easier to compare model runs over time without a
cloud account. Tracking uses a local SQLite database at `mlflow.db`; run
artifacts are stored in the project-root `mlartifacts/` directory. Both are
ignored by Git. Running the tracker again creates new runs and keeps earlier
experiments.

In MLflow:

- An **experiment** is a named collection of related model trials.
- A **run** is one execution of one candidate model and its recorded results.
- A **parameter** is a setting, such as the random seed or tree count.
- A **metric** is a measured number, such as cross-validation ROC-AUC.
- An **artifact** is a file saved with a run, such as a model or plot.
- The **tracking URI** tells MLflow where it stores experiment data. Here it is
  a local SQLite database, with model and file artifacts stored in `mlartifacts/`.

`src/heart_disease/track_experiments.py` reuses the Step 4 split,
preprocessing/model pipelines, classifier settings, cross-validation setup,
and held-out evaluation approach. It compares Logistic Regression and Random
Forest, selects by mean ROC-AUC from five-fold stratified CV on training rows,
and only then logs held-out test results. Runs include hyperparameters,
training and CV metrics, test metrics, a serialized scikit-learn pipeline, the
Step 4 evaluation JSON, and the model's confusion-matrix plot. The tracker also
writes `reports/mlflow_tracking_summary.json`. MLflow stores the fitted
pipelines using its `cloudpickle` serialization format; only load model
artifacts from a source you trust.

### Run Step 5 from Windows PowerShell

Start in the project root and check whether the virtual environment exists:

```powershell
Get-Location
Test-Path .\.venv\Scripts\python.exe
```

This workspace now has a project-local `.venv` created with Python 3.14. If the
environment is absent in your copy, create one with an installed Python version
(this workspace previously reported Python 3.14):

```powershell
py -3.14 -m venv .venv
```

Use the environment's Python directly. This avoids PowerShell execution-policy
errors that can occur when running `Activate.ps1`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set the Python module search path and run the tracker:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m heart_disease.track_experiments
```

Running the tracker creates new MLflow runs. If you only want to inspect
existing runs, do not run the tracker; start the UI with the command below.

`$env:PYTHONPATH` is required because the `heart_disease` package is inside
`src/`. The assignment ` $env = "$PWD\src" ` does not set a PowerShell
environment variable and will not make Python find the package.

After tracking completes, open a second PowerShell terminal in the project
root and start MLflow's local interface with the same environment:

```powershell
$trackingDb = (Join-Path $PWD "mlflow.db") -replace "\\", "/"
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///$trackingDb" --host 127.0.0.1 --port 5000
```

Visit <http://127.0.0.1:5000>. Choose the
`heart-disease-classification` experiment to see its runs. Open a run to view
parameters, metrics, and artifacts; select multiple runs to compare them.

The tracker measures its results at runtime. The first recorded model choice
uses the training-only CV ROC-AUC rule described in Step 4; held-out scores are
for final reporting only and are not used to tune the models. Baseline metrics
on this small dataset are educational, not evidence of clinical performance.

## Step 6: model packaging and reproducible prediction

Step 4 selected Logistic Regression because it had the higher mean five-fold
training-only CV ROC-AUC. The fitted winner is already stored, together with
its fitted preprocessing steps, in `artifacts/selected_model.joblib`. Step 6
does not retrain it or replace that file. Instead,
`src/heart_disease/package_model.py` validates the existing model and report,
then makes a separate copy at
`artifacts/final_heart_disease_pipeline.joblib`. The script refuses to overwrite
different existing output files; if both outputs already match the selected
source, it verifies and reuses them.

The matching `artifacts/final_heart_disease_metadata.json` records the selected
model, actual Step 4 CV and test metrics, the feature schema and target mapping,
package versions, the artifact SHA-256, and a reference to a matching successful
Step 5 run when one is available. The pipeline keeps median imputation and
standard scaling for numeric features, most-frequent imputation and one-hot
encoding for categorical features, and the Logistic Regression classifier
together. Prediction loads this fitted pipeline; it does not fit or refit
preprocessing.

The existing model was serialized with scikit-learn 1.9.0, while this project's
current environment has scikit-learn 1.9.1. Loading emits scikit-learn's
version-mismatch warning. The packaged model produced consistent predictions
across separate fresh processes in this environment, but that test does not
guarantee compatibility with every scikit-learn version. The metadata records
both versions; use the recorded serialization version when you need the closest
environment match.

The model expects all 13 features in the order used by the training CSV:
`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`,
`oldpeak`, `slope`, `ca`, `thal`. Numeric features accept finite JSON numbers.
Coded categorical features accept the documented integer codes; `null` is
accepted for a missing value so the saved imputers can handle it. Do not send
the target column `num` as an input.

Running `python -m heart_disease.predict` with no input uses the first row's
13 features from the cleaned Cleveland CSV as a **demonstration only**. The
script does not read or use that row's target and does not score it as an
evaluation example. For an actual new example, pass a JSON file with all 13
features using `--input-file path\to\example.json`, or pass a JSON object with
`--input-json`. The model outputs class `0` (absence) or `1` (presence); where
supported, it also returns estimated class probabilities. These model scores
are not necessarily calibrated, are not a medical diagnosis, and must not be
used as medical advice.

### Run Step 6 in Windows PowerShell

Run each command from the `heart-disease-mlops` project root, in order. The
existing virtual environment is `.venv`.

**Step A — open PowerShell at the project root and confirm it:**

```powershell
Set-Location 'C:\Users\hemak\Documents\koushikaa-2025ae05628\2025ae05628_mlos\heart-disease-mlops'
Get-Location
```

**Step B — use the existing environment's Python directly:**

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

**Step C — install the listed project dependencies:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Step D — expose the `src` package directory to Python:**

```powershell
$env:PYTHONPATH = "$PWD\src"
```

PowerShell requires the `$env:PYTHONPATH` form; `$env = "$PWD\src"` does not
set Python's module search path.

**Step E — package the existing selected fitted pipeline:**

```powershell
.\.venv\Scripts\python.exe -m heart_disease.package_model
```

**Step F — run a demonstration prediction:**

```powershell
.\.venv\Scripts\python.exe -m heart_disease.predict
```

To predict an actual new example instead, put one JSON object containing all
13 features in a UTF-8 file and run:

```powershell
.\.venv\Scripts\python.exe -m heart_disease.predict --input-file .\my_example.json
```

The same JSON object may also be passed inline with `--input-json`. For a
repeatable local verification, run the prediction command twice with the same
input and compare the predicted class and scores. Do not rerun Step 4 training
or Step 5 tracking just to package or predict.

## Step 7: automated tests and CI

Run the project's tests locally from its root in PowerShell. These commands
call the project virtual environment's Python directly, so activation policy
does not prevent them from using `.venv`:

```powershell
Get-Location
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest tests\ -v
.\.venv\Scripts\python.exe -m ruff check .
git status
```

`requirements-dev.txt` contains pytest and Ruff so test/lint tools are kept
separate from runtime dependencies. `pyproject.toml` configures Ruff to check
common Python errors, unused imports, and import ordering. Ruff is configured
for the whole project; CI fails if it reports an issue.

The local workflow configuration at `.github/workflows/mlops-ci.yml` is set to
run for pushes and pull requests. It uses Python 3.11, installs both
requirement files, sets `PYTHONPATH` to the repository's `src` directory, runs
Ruff, and then runs the pytest suite. This configuration is not evidence of a
successful remote run. At the final documentation check, GitHub Actions
reported no workflows or runs for the remote repository; the project files in
this workspace are untracked, so they have not yet been committed or pushed.

Generated data, the fitted models, and MLflow files are excluded by
`.gitignore`, so GitHub Actions does not receive those local files. The
preprocessing tests use the actual Cleveland column schema and observed example
rows and can run without the full dataset. The saved-model integration test
runs locally when the generated artifacts and cleaned dataset are present; if
they are absent in a clean CI checkout, pytest reports that integration test as
skipped rather than treating missing local files as a passing model test.

To review and push your work, first inspect `git status` and the diffs in VS
Code's Source Control view. Stage and commit only the intended Step 7 files.
Then push your branch yourself; no changes are pushed automatically by this
project. On GitHub, open the repository's **Actions** tab, select the
**MLOps CI** workflow, and open the run for your push or pull request. Confirm
the Ruff and pytest steps both show a green check. Open a failed step's logs
for its error details.

## Step 8: Docker containerization

The `Dockerfile` builds an inference image containing the packaged model and
metadata. The image uses the same pinned scikit-learn version that serialized
the saved pipeline. It excludes datasets, tests, notebooks, and local MLflow
files. The current image starts the FastAPI service with Step 10 monitoring enabled.
The final documentation verification image is tagged `step11`. The prediction
CLI remains available with
`docker run --rm heart-disease-mlops:step11 python -m heart_disease.predict
--help`.

## Step 9: Production prediction API

The FastAPI service in `src/heart_disease/api.py` loads the existing fitted
pipeline from `artifacts/final_heart_disease_pipeline.joblib` and caches it in
the API process. It does not retrain the model or fit preprocessing. The
pipeline was selected as Logistic Regression. The API accepts the following
13 input features in the saved model's order:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`,
`oldpeak`, `slope`, `ca`, `thal`.

Continuous features (`age`, `trestbps`, `chol`, `thalach`, and `oldpeak`) accept
JSON numbers. The other features are integer category codes matching the
project's UCI Cleveland schema. A feature may be `null` when unknown; the saved
preprocessing pipeline applies the imputers fitted during training. Missing,
extra, non-numeric, or unsupported category values are rejected with HTTP 422.
Target class `0` means absence and `1` means presence (original UCI `num`
values 1–4). Prediction scores are model estimates, may not be calibrated, and
are not a medical diagnosis or advice.

### Run locally in Windows PowerShell

Run these commands from the project root. Keep the API server running in this
terminal; use another terminal for requests or tests. These commands use the
project environment directly and do not require activating it:

```powershell
Get-Location
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m uvicorn heart_disease.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open the interactive API documentation at
`http://127.0.0.1:8000/docs`. `GET /health` returns `{"status":"ok"}`. Use
`POST /predict` and send the feature object shown below. This is a feature-only
demonstration row observed in the project's Cleveland data; it is not an
evaluation example or medical guidance. If port 8000 is occupied, change `--port 8000` to an available port, such as
`--port 18001`, and use that port in the URLs and request below.

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

You can also send that request from a second PowerShell terminal:

```powershell
$body = @{
    age = 63; sex = 1; cp = 1; trestbps = 145; chol = 233
    fbs = 1; restecg = 2; thalach = 150; exang = 0
    oldpeak = 2.3; slope = 3; ca = 0; thal = 6
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' `
    -Method Post -ContentType 'application/json' -Body $body
```

### Test locally

Stop the local server with `Ctrl+C`, then run from the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -v
.\.venv\Scripts\python.exe -m pytest tests\ -v
.\.venv\Scripts\python.exe -m ruff check .
```

The API integration test uses the local saved artifact and cleaned dataset. If
those Git-ignored files are unavailable (for example, in a clean checkout), the
test reports a skip; it does not retrain or create replacement data.

### Build and run the API container

Start Docker Desktop and confirm `docker version` shows both Client and Server.
From the project root, build the updated image:

```powershell
Get-Location
docker build -t heart-disease-mlops:step11 .
```

Run the API container in a terminal:

```powershell
docker run --rm -p 127.0.0.1:18001:8000 --name heart-disease-api heart-disease-mlops:step11
```

While it is running, open `http://127.0.0.1:18001/docs` or use the PowerShell
request above from a second terminal, changing its URL to port 18001. Choose
another available host port if 18001 is occupied. Stop the container with
`Ctrl+C`.

## Step 10: Monitoring and model performance tracking

The API exposes Prometheus-formatted operational metrics at
`http://127.0.0.1:8000/metrics` (or the published container port). The endpoint
is scrapeable by a Prometheus server, but this project does not install or
configure a separate server, dashboard, cloud service, or alerting system.

Metrics include:

- `heart_disease_api_requests_total` — request totals by HTTP method, known
  route, and status code.
- `heart_disease_predictions_succeeded_total` — successful predictions.
- `heart_disease_api_request_failures_total` — failed requests by known route,
  status, and bounded reason.
- `heart_disease_api_request_duration_seconds` — request latency histogram.
- `heart_disease_input_feature_issues_total` — missing, null, invalid category,
  invalid request value, and out-of-training-range feature observations.

The API emits one JSON log line per completed request with a UTC timestamp,
request ID, route, status, latency, and (on success) predicted class and label.
It deliberately does not log request bodies, feature values, probabilities,
client addresses, or validation error text. Metrics and request counters are
held in process memory; they reset when the API process/container restarts.

Input monitoring uses the schema's documented integer category codes. Numeric
monitoring references are the inclusive observed minima and maxima in the
project's exact 242-row stratified training split (random seed 42):

| Feature | Training minimum | Training maximum |
| --- | ---: | ---: |
| `age` | 29 | 77 |
| `trestbps` | 94 | 200 |
| `chol` | 126 | 564 |
| `thalach` | 71 | 202 |
| `oldpeak` | 0.0 | 6.2 |

Out-of-training-range values are counted, not rejected. These are empirical
training-data boundaries—not clinical limits, proof of invalidity, or a drift
threshold. Nulls and invalid category/input values are counted without their
submitted values being logged. The configured imputers still handle permitted
null feature values as before.

The API does not receive ground-truth labels, so it cannot measure live
accuracy, model degradation, or label drift. If verified labels are collected
later under an appropriate privacy/consent process, evaluate an independently
labelled dataset offline: validate the same 13 features, map the original UCI
target using `num == 0 -> 0` and `num in 1..4 -> 1`, run the frozen packaged
pipeline, and calculate accuracy, precision, recall, and ROC-AUC against those
labels. Compare results only across comparable, labelled cohorts and define
any alert thresholds before interpreting performance. The current project
reports no such production evaluation.

## Step 11: Deploy locally to Kubernetes and view a Grafana dashboard

The canonical manifests are `k8s/deployment.yaml` and `k8s/service.yaml`;
`k8s/monitoring.yaml` configures Prometheus and Grafana. The older combined
manifest at `deployment/kubernetes/heart-disease-stack.yaml` is retained for
history; use the split `k8s/` manifests below. The Grafana dashboard displays
request rate, successful predictions, HTTP failure ratio, and p95 latency.
API logs remain privacy-safe JSON on container stdout/stderr and can be viewed
with `kubectl logs`.

This is a **local Docker Desktop Kubernetes deployment**, not a public-cloud
deployment. It requires Docker Desktop with Kubernetes enabled and a ready
`docker-desktop` kubectl context. The API Service is `LoadBalancer`; Docker
Desktop assigns a private load-balancer IP, not a public endpoint. The
assignment-required Service type is kept as `LoadBalancer`; on this host,
`kubectl port-forward` is used to access the service reliably from Windows. Do
not expose the demo Grafana service publicly. Grafana's default demo
credentials are `admin` / `admin`; this is for local coursework only, and you
should change the password before using the dashboard beyond this private
local environment.

Run these commands from the project root in PowerShell:

```powershell
docker version
kubectl get contexts
kubectl config current-context
kubectl get nodes
docker build -t heart-disease-mlops:step11 .
kubectl apply --dry-run=client -f k8s\deployment.yaml
kubectl apply --dry-run=client -f k8s\service.yaml
kubectl apply --dry-run=client -f k8s\monitoring.yaml
kubectl apply -f k8s\monitoring.yaml
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=300s
kubectl rollout status deployment/prometheus -n heart-disease --timeout=300s
kubectl rollout status deployment/grafana -n heart-disease --timeout=300s
kubectl get pods,svc -n heart-disease -o wide
```

If the context is not `docker-desktop`, select it in Docker Desktop's
Kubernetes settings or run `kubectl config use-context docker-desktop` after
confirming that context exists. The first apply downloads Prometheus and
Grafana images and may take several minutes. Docker Desktop's Kubernetes
container runtime can pull the locally tagged API image; `imagePullPolicy:
IfNotPresent` also allows reuse of an image already present on the cluster.

To troubleshoot startup, run:

```powershell
kubectl get pods -n heart-disease -o wide
kubectl describe pods -n heart-disease
kubectl get events -n heart-disease --sort-by=.lastTimestamp
kubectl logs -n heart-disease deployment/heart-disease-api --tail=100
kubectl logs -n heart-disease deployment/prometheus --tail=100
kubectl logs -n heart-disease deployment/grafana --tail=100
kubectl get endpoints -n heart-disease
```

For a reliable Windows-local connection, open two additional PowerShell
terminals at the project root and keep each port-forward running:

**Terminal 2 — API:**

```powershell
kubectl port-forward -n heart-disease service/heart-disease-api 18002:8000
```

**Terminal 3 — Grafana:**

```powershell
kubectl port-forward -n heart-disease service/grafana 13000:3000
```

Grafana is a browser dashboard, not a container shown in Docker Desktop's
regular Containers list. In this Docker Desktop setup, its Kubernetes
`ClusterIP` service is not directly reachable from Windows; keep this
port-forward terminal running while using the dashboard. If it closes, rerun
the command. Starting the forward automatically after Windows sign-out or
restart would require a separate Windows service or scheduled task.

Open `http://127.0.0.1:13000`, sign in with `admin` / `admin`, then change the
password. Open the **Heart Disease API** dashboard. In another PowerShell
terminal, check the API and generate a sample prediction:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:18002/health'
$body = @{
    age = 63; sex = 1; cp = 1; trestbps = 145; chol = 233
    fbs = 1; restecg = 2; thalach = 150; exang = 0
    oldpeak = 2.3; slope = 3; ca = 0; thal = 6
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:18002/predict' `
    -Method Post -ContentType 'application/json' -Body $body
Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:18002/metrics'
```

Prometheus scrapes every 15 seconds, so allow at least one scrape interval
after generating requests before expecting dashboard points. The API request
logs can be read with:

```powershell
kubectl logs -n heart-disease deployment/heart-disease-api --tail=50
```

Capture your remaining assignment screenshots yourself while these services
are running. Keep screenshots in `screenshots/kubernetes/` and
`screenshots/monitoring/`; the current Grafana dashboard screenshot is saved
at `screenshots/monitoring/grafana-dashboard.png`. To capture the remaining
evidence, use Windows Snipping Tool (`Win+Shift+S`) and include:

1. `kubectl get nodes` showing the ready local node.
2. `kubectl get deployment,pods -n heart-disease -o wide` showing ready API,
   Prometheus, and Grafana workloads.
3. `kubectl get service -n heart-disease -o wide` and the API port-forward
   terminal.
4. `http://127.0.0.1:18002/docs` showing Swagger UI.
5. `http://127.0.0.1:18002/health` showing the successful response.
6. A successful `/predict` response in Swagger or PowerShell.
7. The Grafana dashboard at `http://127.0.0.1:13000/d/heart-disease-api/heart-disease-api`
   with live data.

The saved Grafana image is actual dashboard evidence; the other listed
screenshots still need to be captured manually if the assignment expects each
individual view.

To stop local access, press `Ctrl+C` in each port-forward terminal. To remove
the local demo stack later, run
`kubectl delete -f k8s\service.yaml`, then
`kubectl delete -f k8s\deployment.yaml`, then
`kubectl delete -f k8s\monitoring.yaml`. This
removes the Kubernetes namespace and its demo resources; it does not delete
the local model, source dataset, or MLflow files.

A captured dashboard view is available at
[`screenshots/monitoring/grafana-dashboard.png`](screenshots/monitoring/grafana-dashboard.png).

### Run and inspect monitoring locally

Run these from the project root. These commands use the project's Python
directly, so PowerShell script-activation policy does not block them:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m uvicorn heart_disease.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open `http://127.0.0.1:8000/metrics` in a browser. The endpoint displays raw
Prometheus exposition text. Request and prediction logs appear in the same
PowerShell terminal running Uvicorn. To run the monitoring and full test suite
from a second terminal at the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest tests\test_monitoring.py tests\test_api.py -v
.\.venv\Scripts\python.exe -m pytest tests\ -v
.\.venv\Scripts\python.exe -m ruff check .
```

### Run monitoring in Docker

Build the local API image and start the API:

```powershell
docker build -t heart-disease-mlops:step11 .
docker run --rm -p 127.0.0.1:18001:8000 --name heart-disease-api heart-disease-mlops:step11
```

View `http://127.0.0.1:18001/metrics` for metrics and
`http://127.0.0.1:18001/docs` for API documentation. JSON request logs are
printed to the container terminal. In another PowerShell terminal, run the
tests on the host with the commands above. No training or model artifact
changes are required for monitoring.
