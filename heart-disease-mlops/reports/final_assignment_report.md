# Heart Disease MLOps Assignment — Final Report

- **Verification date:** 2026-10-20
- **Project:** `heart-disease-mlops`
- **Scope:** Local project review, experiment evidence, tests, Docker build,
  Docker Desktop Kubernetes deployment, API checks, and Grafana monitoring
  verification. No public-cloud deployment or GitHub push was performed.

## Executive summary

The workspace contains a local end-to-end educational MLOps pipeline using the
processed Cleveland Heart Disease dataset from UCI. It includes data
acquisition and validation, a train-fold-only preprocessing pipeline, model
comparison, local MLflow tracking, a saved fitted pipeline, pytest/Ruff,
a GitHub Actions workflow file, a Docker image, a FastAPI prediction service,
and operational Prometheus metrics plus privacy-conscious request logs.

At this verification, the local pytest suite passed (15 tests), Ruff passed,
dependency consistency passed, and the Step 11 Docker image built and ran on
Docker Desktop Kubernetes. The API, Prometheus scrape, Grafana dashboard, and
structured logs were verified. GitHub Actions is
**configured in the local workspace but not verified as a remote run**:
GitHub returned no workflows/runs for the repository and the working files are
untracked in the local Git worktree.

## Implemented pipeline and architecture

```mermaid
flowchart LR
    UCI[UCI processed Cleveland data] --> ACQ[Acquire, validate, clean]
    ACQ --> SPLIT[Stratified 80/20 split, seed 42]
    SPLIT --> CV[5-fold CV on training split]
    CV --> SELECT[Select by mean CV ROC-AUC]
    SELECT --> TEST[One held-out test evaluation]
    CV --> MLFLOW[MLflow: SQLite metadata + mlartifacts]
    TEST --> REPORT[Evaluation JSON and confusion matrices]
    SELECT --> PACKAGE[Existing fitted pipeline + metadata]
    PACKAGE --> API[FastAPI /predict]
    API --> HEALTH[/health]
    API --> METRICS[/metrics and privacy-safe logs]
    IMAGE[Docker image] --> API
    CODE[pytest + Ruff] --> CI[GitHub Actions workflow configured]
    API --> LB[Local Kubernetes LoadBalancer Service]
    PROM[Prometheus scrape] --> API
    PROM --> GRAF[Grafana dashboard]
```

### Dataset, features, target, and preprocessing

- **Dataset:** UCI Machine Learning Repository, processed Cleveland subset,
  `processed.cleveland.data`; local file:
  `data/raw/processed.cleveland.data`. Dataset notes and source links are in
  [`data/raw/README.md`](../data/raw/README.md).
- **Size:** 303 rows and 14 source columns (13 input features plus `num`).
- **Source target:** `num=0` means absence; `num=1..4` means presence. Model
  target is binary: `0=absence`, `1=presence`.
- **Feature order:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`,
  `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
- **Numerical features:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`;
  median imputation followed by standard scaling.
- **Categorical integer-code features:** `sex`, `cp`, `fbs`, `restecg`,
  `exang`, `slope`, `ca`, `thal`; most-frequent imputation followed by
  one-hot encoding.
- Preprocessing is inside each scikit-learn pipeline, fitted within CV folds
  and on the final training split. The saved prediction pipeline is already
  fitted; prediction does not fit preprocessing.

### Model selection and measured results

The logged candidates are Logistic Regression and Random Forest. Selection
used the highest mean ROC-AUC from five stratified folds on training data only;
the held-out test set did not select or tune the model. The values below come
from `reports/model_evaluation.json`,
`reports/mlflow_tracking_summary.json`, and matching finished local MLflow
runs. They are educational results on a small dataset, not clinical evidence.

**Training-only five-fold cross-validation (mean ± sample standard deviation):**

| Model | Accuracy | Precision (presence) | Recall (presence) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8471 ± 0.0112 | 0.8768 ± 0.0606 | 0.7834 ± 0.0513 | **0.9025 ± 0.0161** |
| Random Forest | 0.8139 ± 0.0259 | 0.7967 ± 0.0298 | 0.8008 ± 0.0898 | 0.8936 ± 0.0372 |

**Held-out test split (61 rows):**

| Model | Accuracy | Precision (presence) | Recall (presence) | F1 (presence) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (selected) | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9665 |
| Random Forest | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9443 |

Confusion matrices use label order `[absence, presence]` for both rows and
columns:

- Logistic Regression: `[[28, 5], [2, 26]]`
- Random Forest: `[[27, 6], [2, 26]]`

The selected model is Logistic Regression. The packaged artifact is
`artifacts/final_heart_disease_pipeline.joblib`; metadata is
`artifacts/final_heart_disease_metadata.json`. The recorded model artifact
SHA-256 is
`3432bdf59ce436280d5084d20d97f2254c5515163712c64ef096a66f454f370c`.
It matches the existing `artifacts/selected_model.joblib`; packaging reused
the fitted model rather than retraining it.

### MLflow evidence

The actual local tracking database is `mlflow.db`. Experiment
`heart-disease-classification` has experiment ID `1`, with artifact location
under the project `mlartifacts/` directory. Direct inspection of this
database via `MlflowClient` found **5 runs: 4 FINISHED and 1 FAILED**.

| Run ID | Model | Status | Training CV ROC-AUC | Test accuracy | Test precision | Test recall | Test ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dd5b5052f3f549c3a8e46d3cf392ae19` | Logistic Regression | FAILED | 0.9025 | 0.8852 | 0.8387 | 0.9286 | 0.9665 |
| `860c5a6bda3b4d81abdc954cb6deea41` | Logistic Regression | FINISHED | 0.9025 | 0.8852 | 0.8387 | 0.9286 | 0.9665 |
| `fb6bfb53d35343bfad4d638619facf8e` | Random Forest | FINISHED | 0.8936 | 0.8689 | 0.8125 | 0.9286 | 0.9443 |
| `b0d81c076d1f4aa0b803b3c7e4b0df34` | Logistic Regression | FINISHED | 0.9025 | 0.8852 | 0.8387 | 0.9286 | 0.9665 |
| `d81ef620515240d7b0c686d9a3abb2c4` | Random Forest | FINISHED | 0.8936 | 0.8689 | 0.8125 | 0.9286 | 0.9443 |

Each finished run has logged parameters and metrics. The failed run also
contains metrics from the run's work before it failed; it is not counted as a
successful run. The selected-run reference recorded in the packaged metadata
is `b0d81c076d1f4aa0b803b3c7e4b0df34`. No run was deleted or changed for this
report.

To display the experiment locally from the project root:

```powershell
$trackingDb = (Join-Path $PWD 'mlflow.db') -replace '\\', '/'
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///$trackingDb" --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and select `heart-disease-classification`.

## Tests, CI, Docker, API, and monitoring evidence

- **Local tests:** `15 passed` in the final suite; the saved-model tests used
  local generated artifacts/data. No production data is required by the
  monitoring unit tests.
- **Warnings:** the local environment uses scikit-learn 1.9.1 while the model
  was serialized with 1.9.0. The existing tests report
  `InconsistentVersionWarning`. The Docker requirements pin scikit-learn 1.9.0.
- **Lint/dependencies:** `python -m ruff check .` passed; `python -m pip check`
  reported no broken requirements.
- **CI configuration:** `.github/workflows/mlops-ci.yml` configures pushes and
  pull requests to use Python 3.11, install runtime and dev requirements, set
  `PYTHONPATH`, run Ruff, and run pytest. **Remote CI status: NOT VERIFIED.**
  GitHub Actions API queries returned zero workflows and a 404 for runs of
  `mlops-ci.yml`. The local source files are currently untracked; no push was
  made.
- **Docker:** `heart-disease-mlops:step11` built successfully. A container
  smoke test on host port 18001 verified `/health` (HTTP 200 and `status=ok`),
  `/docs` (HTTP 200), `/metrics` (HTTP 200 and Prometheus output), a valid
  `POST /predict` response, and invalid-category HTTP 422. The container was
  stopped after verification. The earlier Step 10 smoke test also verified
  structured request logs did not contain raw input feature values.
- **Port note:** host port 8001 was already occupied by a Python process
  (PID 28712), so it was not stopped or changed. Docker verification used
  available port 18001 instead.
- **API:** `GET /health`, `POST /predict`, `GET /metrics` (Prometheus
  exposition), and Swagger UI at `/docs`. Local ports may be occupied; choose
  another available port if the example port is busy.
- **Local Kubernetes deployment:** the `docker-desktop` context was active and
  its node was Ready. The API, Prometheus, and Grafana deployments each reached
  1/1 Ready. The API Service is `LoadBalancer`, with local cluster IP
  `172.18.0.5` on port 8000 (not a public endpoint); Windows-host verification
  used `kubectl port-forward` because the cluster IP was not directly reachable
  from the host. The existing Docker image was used; no model training or
  artifact regeneration occurred.
- **Kubernetes API verification:** `/health` returned `{"status":"ok"}`,
  `/metrics` returned HTTP 200 with the expected Prometheus counters, and a
  valid `POST /predict` returned the saved-model response. Prometheus reported
  target `heart-disease-api:8000` as `up=1`; the success counter was observed
  as 1 after the sample prediction. Grafana's API returned the provisioned
  `Heart Disease API` dashboard with four panels. A live dashboard screenshot
  is saved at `screenshots/monitoring/grafana-dashboard.png`.
- **Monitoring/logging:** Prometheus scrapes every 15 seconds and Grafana
  charts request rate, successful predictions, the failed-request counter, p95 request
  latency, and feature-quality issues. API request logs are JSON on container
  stdout and accessible with `kubectl logs`; they exclude raw input values.
  Prometheus history is local and ephemeral; no persistent volume, alerting,
  or public monitoring endpoint is configured.
- **Screenshots/video:** the Grafana monitoring screenshot is present at
  `screenshots/monitoring/grafana-dashboard.png`. Individual screenshots for
  the cluster, pods/services, docs, health response, and prediction response
  still need to be captured manually. No video demonstration or public
  deployment URL exists.

## Limitations and future improvements

1. This model is an educational baseline trained/evaluated on only 303
   Cleveland observations and must not be used to make medical decisions.
2. The held-out evaluation is one split, not evidence of clinical validity or
   performance on a new population.
3. Live ground-truth labels are not available, so production accuracy,
   degradation, and label drift are **not measured**. If independently
   labelled data is later collected appropriately, map `num=0` to class 0 and
   `num=1..4` to class 1, run the frozen pipeline, and calculate metrics on
   comparable labelled cohorts. Choose thresholds before interpreting alerts.
4. The local Kubernetes demo has no persistent storage, so Prometheus history
   and Grafana state may be lost when pods are recreated. Configure persistence
   before using this setup for longer-term monitoring.
5. Local model artifacts and source datasets are ignored by Git. Reproducible
   builds from a clean checkout need those approved model artifacts to be
   provisioned locally before Docker build; do not commit large or sensitive
   data without project approval.
6. Add real model/artifact provisioning and exercise GitHub Actions after the
   changes are committed/pushed. Capture a video only if the assignment rubric
   requires one.
7. Resolve the Python package environment warning by installing the exact
   serialized scikit-learn version when using the saved joblib artifact.

## Assignment deliverables checklist

| Deliverable | Status | File or evidence |
| --- | --- | --- |
| Dataset source and schema documented | Complete locally | `data/raw/README.md`; `README.md` |
| Data acquisition, cleaning, and validation | Implemented | `src/heart_disease/data_acquisition.py`, `src/heart_disease/data_preprocessing.py` |
| EDA outputs | Present locally | `src/heart_disease/eda.py`, `reports/figures/` |
| Reproducible split and preprocessing | Implemented | `src/heart_disease/feature_engineering.py` |
| Candidate model comparison and evaluation | Evidence present | `src/heart_disease/train.py`, `src/heart_disease/evaluate.py`, `reports/model_evaluation.json` |
| MLflow parameters, metrics, and run history | Local store verified | `src/heart_disease/track_experiments.py`, `mlflow.db`, `mlartifacts/` |
| Packaged selected fitted model and metadata | Present locally | `artifacts/final_heart_disease_pipeline.joblib`, `artifacts/final_heart_disease_metadata.json` |
| Prediction CLI and fitted preprocessing reuse | Implemented/tested | `src/heart_disease/predict.py`, `tests/test_saved_model.py` |
| Preprocessing/API/monitoring tests | Local PASS (15 total) | `tests/` |
| CI workflow configuration | Present locally; remote run unverified | `.github/workflows/mlops-ci.yml` |
| Docker image and API smoke test | Local PASS | `Dockerfile`; `heart-disease-mlops:step11` |
| Kubernetes deployment manifest and health probes | Local PASS | `k8s/deployment.yaml`; Docker Desktop |
| API exposed using Kubernetes LoadBalancer Service | Local PASS (private local cluster) | Service `heart-disease-api`, cluster IP `172.18.0.5:8000`; no public endpoint |
| FastAPI health, prediction, and docs routes | Local PASS | `src/heart_disease/api.py`; `/health`, `/predict`, `/docs` |
| Prometheus scrape and Grafana dashboard | Local PASS | `k8s/monitoring.yaml`; `screenshots/monitoring/grafana-dashboard.png` |
| Operational metrics and privacy-safe request logs | Local PASS | `src/heart_disease/monitoring.py`; `/metrics`; `kubectl logs` |
| Ground-truth based production performance monitoring | Not measured | No production labels; no live performance data |
| GitHub Actions successful run | Not verified | Remote reports no workflow/run; push not performed |
| Deployment/monitoring screenshot | Partial | Grafana at `screenshots/monitoring/grafana-dashboard.png`; capture remaining Kubernetes/API views manually |
| Video demonstration | Not found | No video file present |
| Public cloud deployment URL | Not implemented / not requested | Local Docker Desktop Kubernetes selected instead |

### Section 7 — Production deployment checklist

| Requirement | Status | Actual evidence |
|---|---|---|
| Dockerized API deployed to local Kubernetes | **PASS** | `docker-desktop`; `heart-disease-api` Pod Ready 1/1 |
| Deployment manifest present and applied | **PASS** | `k8s/deployment.yaml`; server dry-run and apply succeeded |
| Required Service exposure type | **PASS** | `k8s/service.yaml`, `type: LoadBalancer`; private IP `172.18.0.5`, Windows access via port-forward |
| `/health`, `/docs`, `/predict` verified | **PASS** | Kubernetes API via local port-forward; successful health and prediction; `/docs` status 200 |
| Deployment screenshots available | **PARTIAL** | Grafana screenshot present; capture cluster/pods/service/API endpoint views manually |

### Section 8 — Monitoring and logging checklist

| Requirement | Status | Actual evidence |
|---|---|---|
| API request logging works | **PASS** | `kubectl logs -n heart-disease deployment/heart-disease-api`; JSON request entries |
| Request counts, errors, and latency metrics work | **PASS** | `/metrics`; counters include test GET/POST and HTTP 422; latency histogram available |
| Prometheus and Grafana integration works | **PASS** | `k8s/monitoring.yaml`; all deployments Ready; provisioned dashboard has four panels |
| Dashboard shows actual deployed API data | **PASS** | Prometheus target query `up{job="heart-disease-api"}=1`; dashboard screenshot saved |
| Monitoring screenshot available | **PASS** | `screenshots/monitoring/grafana-dashboard.png` |

No assignment rubric file was present in the repository during this check.
Confirm this checklist against any separate instructor-provided rubric.

## Final local verification commands (Windows PowerShell)

From the project root, use the virtual environment's Python directly; this
avoids PowerShell script activation restrictions. Each command below is
separate. Do not retrain or rerun MLflow tracking just to verify the saved
model/API.

```powershell
Set-Location 'C:\Users\hemak\Documents\koushikaa-2025ae05628\2025ae05628_mlos\heart-disease-mlops'
Get-Location
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest tests\ -v
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
docker version
docker image ls heart-disease-mlops
docker build -t heart-disease-mlops:step11 .
kubectl config current-context
kubectl get nodes
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

In separate PowerShell terminals, port-forward the API and Grafana:

```powershell
kubectl port-forward -n heart-disease service/heart-disease-api 18002:8000
```

```powershell
kubectl port-forward -n heart-disease service/grafana 13000:3000
```

Then verify API and Grafana:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:18002/health'
Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:18002/metrics'
Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:13000/api/health' -Headers @{ Authorization = 'Basic YWRtaW46YWRtaW4=' }
$body = @{ age = 63; sex = 1; cp = 1; trestbps = 145; chol = 233; fbs = 1; restecg = 2; thalach = 150; exang = 0; oldpeak = 2.3; slope = 3; ca = 0; thal = 6 } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:18002/predict' -Method Post -ContentType 'application/json' -Body $body
kubectl logs -n heart-disease deployment/heart-disease-api --tail=50
```

The Grafana dashboard is at `http://127.0.0.1:13000/d/heart-disease-api/heart-disease-api`.
The demo Grafana login is `admin` / `admin`; change it before any broader use.
The output of:

```powershell
docker run --detach --rm --name heart-disease-final -p 127.0.0.1:18001:8000 heart-disease-mlops:step11
Start-Sleep -Seconds 5
Invoke-RestMethod -Uri 'http://127.0.0.1:18001/health'
Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:18001/metrics'
Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:18001/docs'
$body = @{ age = 63; sex = 1; cp = 1; trestbps = 145; chol = 233; fbs = 1; restecg = 2; thalach = 150; exang = 0; oldpeak = 2.3; slope = 3; ca = 0; thal = 6 } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:18001/predict' -Method Post -ContentType 'application/json' -Body $body
docker stop heart-disease-final
git status --short --branch
```

Port 8001 was occupied by an existing Python process during this check. The
host port 18001 was used for the verified Docker run; if it is occupied later,
choose a different available host port and use it consistently in the URL.
The final `git status` is for reviewing; stage, commit, and push only after
your approval.
