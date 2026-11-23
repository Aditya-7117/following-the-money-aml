# Following the Money: Detecting Money Laundering from Account Transaction Patterns

Group project for CSC1181, Foundations of Statistical Analysis and Machine Learning, Dublin City University, 2026.

Detecting money laundering in IBM's synthetic AMLworld transactions with logistic regression, Naive Bayes and a decision tree, using features built only from each account's earlier activity.

## Data

IBM Transactions for Anti Money Laundering (AML), HI-Small files, from Kaggle:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Download `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` individually and put them in `data/raw/`. The data is not stored in this repository. Licence: CDLA-Sharing-1.0.

Altman, E. et al. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.16424.

## Layout

| Folder | Contents |
|---|---|
| `src/data/` | Loading, features and time splits |
| `src/analysis/` | Statistics and exploratory analysis |
| `src/models/` | Model training and validation |
| `src/evaluation/` | Metrics, error analysis and pattern recall |
| `scripts/` | Command-line entry points |
| `config/` | Feature, split, model and metric settings |
| `tests/` | Automated checks |
| `docs/` | Data dictionary and data source record |
| `artifacts/` | Small recorded outputs, such as check summaries |
| `notebooks/` | Exploration notebooks, one folder per person |

## Quick start

Python 3.12 is supported on macOS, Linux, and Windows (native or WSL). On native Windows, the Unix `resource` module is unavailable, so peak memory telemetry reports `N/A` in terminal summaries and `null` in `manifest.json` without affecting dataset processing or deterministic outputs. Note that native Windows compatibility remains to be confirmed on Windows hardware by teammates.

Download the HI-Small dataset from the source
linked above and place HI-Small_Trans.csv
inside data/raw/.

Run tests:
python3.12 -m unittest discover

Generate a sample:
python3.12 scripts/prepare_data.py --sample

Generate the complete dataset:
python3.12 scripts/prepare_data.py --full

For complete file explanations, output schemas,
reproduction steps and troubleshooting, see
[Data Pipeline Handover](docs/DATA_PIPELINE_HANDOVER.md).