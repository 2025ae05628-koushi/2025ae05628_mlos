# Raw Heart Disease Dataset

## Dataset selected

This project uses the **processed Cleveland** file from the UCI Heart Disease
dataset. It is the original Cleveland subset in comma-separated, headerless
format, published as `processed.cleveland.data`. It contains 303 patient records
and 14 fields (13 input features plus the target). This is not the Hungarian,
Switzerland, Long Beach VA, or Statlog dataset.

- UCI dataset description: <https://archive.ics.uci.edu/dataset/45/heart+disease>
- Exact source file:
  <https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data>
- Field definitions: <https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names>

Run `python src/heart_disease/data_acquisition.py` from the project root to
download this file if it is not already present and print an initial summary.
The script saves the untouched source file here as `processed.cleveland.data`.
It interprets `?` as a missing value only while reading the file for reporting;
it does not alter or rewrite the raw file.

## What one row and each column represent

Each row is one patient's record from the Cleveland subset. The first 13 fields
are input features:

| Column | Meaning |
| --- | --- |
| `age` | Age in years |
| `sex` | Sex: 1 = male, 0 = female |
| `cp` | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure on admission |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar greater than 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiogram result (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression caused by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (1–3) |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia test category (values in this file include 3, 6, and 7) |

`num` is the target (the 14th field), described by UCI as angiographic disease
status. The raw file contains target values `0`, `1`, `2`, `3`, and `4`.
`0` means no diagnosed disease under this dataset's definition (less than 50%
diameter narrowing); `1`–`4` mean disease is present, with the number indicating
the reported extent. The acquisition script reports the target values and their
counts from the downloaded file rather than converting them.

For a future binary prediction task, the common interpretation is `num == 0`
for absence and `num > 0` for presence. No binary conversion or other data
cleaning is performed in this step.

## Git and local files

The raw data file is excluded by the project's `.gitignore` to avoid checking
generated data into Git. This README is explicitly allowed by `.gitignore` so
the source and interpretation notes remain tracked.
