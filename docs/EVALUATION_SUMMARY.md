# Evaluation summary | CSC1181 Group 18

- **Contributor:** Priyanka (evaluation)
- **Branch:** `priyanka/evaluation`
- **Data:** val, test_main and test_tail rows from `data/processed/full_*`, scored with the three saved models in `artifacts/models/`. `data/raw/HI-Small_Patterns.txt` is used only after scoring, to label laundering patterns.
- **Rule:** every model keeps the threshold chosen on validation. Nothing is tuned on test.

## Files

| File | What it does |
|---|---|
| `src/evaluation/metrics.py` | Precision, recall, F1, PR-AUC, ROC-AUC and recall at the 1% alert budget |
| `src/evaluation/patterns.py` | Reads the patterns file and links each record to its `transaction_id` |
| `src/evaluation/evaluate.py` | Scores val and test with the saved models, adds two baselines, builds the tables |
| `scripts/run_evaluation.py` | Runs the whole evaluation and saves the outputs |
| `config/evaluation.json` | Views, alert budget and evaluation rules |
| `tests/evaluation/test_evaluation.py` | Hand-checked unit tests and one full run on made-up data |
| `notebooks/priyanka/evaluation.ipynb` | Results, accuracy check, confusion matrix, PR curves, pattern recall and error analysis |

Outputs are saved in `artifacts/evaluation/`: `results_table.csv`, `budget_table.csv`, `pattern_recall.csv`, `evaluation_summary.json`, `pr_curves.png` and `budget_recall.png`. Every score is also saved in `artifacts/scores/scores.csv`, which is gitignored.

## How to run

1. Prepare the data with `scripts/prepare_data.py --full`, and put the three model files in `artifacts/models/` (or run `scripts/train_models.py`).
2. Run the tests: `python3 -m unittest discover -s tests/evaluation -t .`
3. Run the evaluation: `python3 scripts/run_evaluation.py`
4. Open the notebook and choose Run All.

On Windows, use `python` instead of `python3`.

## Evaluation rules

- `test_main` is the main view (1,014,774 rows, 1,143 laundering, 0.113%). `test_full` adds the 1,108-row tail (59% laundering), so the two views are not independent.
- Baselines: always-legitimate (flags nothing) and an ACH rule (flags every ACH payment).
- Precision with no alerts is written as n/a, never 0.
- On validation, the numbers match `artifacts/models/val_summary.json` exactly.

## Results on test_main

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Recall at 1% budget |
|---|---|---|---|---|---|---|
| Always legitimate | n/a | 0.000 | 0.000 | 0.001 | 0.500 | 0.010 |
| ACH rule | 0.007 | 0.874 | 0.014 | 0.006 | 0.869 | 0.064 |
| Logistic regression | 0.090 | 0.135 | 0.108 | 0.046 | 0.951 | 0.355 |
| Naive Bayes | 0.065 | 0.059 | 0.062 | 0.012 | 0.901 | 0.072 |
| Decision tree | 0.224 | 0.197 | 0.210 | 0.123 | 0.772 | 0.420 |

## What the results show

- The decision tree performs best: PR-AUC 0.123 against 0.0011 for no skill, and 22% of its alerts are real laundering.
- Accuracy hides failure: flagging nothing is 99.89% accurate, more than the decision tree (99.83%).
- PR-AUC is the main metric: logistic regression has the best ROC-AUC (0.95) but a PR-AUC of only 0.046 (Saito and Rehmsmeier, 2015).
- When only the top 1% is reviewed, the decision tree catches 42% of laundering; random checking catches 1%.
- The tree's PR-AUC fell from 0.166 on validation to 0.123 on test, because its settings were tuned on validation.

## Patterns and errors

- All 3,209 pattern records were matched; across the whole dataset, 62% of laundering belongs to a named pattern. On test_main, the tree catches 32% of named-pattern laundering but only 1.5% of laundering with no pattern.
- Every alert from the tree is an ACH payment, so non-ACH laundering is never caught.
- Missed cases often come from senders with no recent activity, and false alarms look like real laundering on all 10 features.

## Note on the 1% budget

The plan broke ties by `transaction_id`. The IDs follow the raw file order, which is not random, so the always-legitimate model appeared to catch 39% of laundering. Tied rows now share the remaining alerts evenly.

## Limitations

- Synthetic data (HI-Small only) and a short test period of about 2.3 days.
- Each result is a single estimate with no confidence interval; with 1,143 laundering cases, small differences between models may not be meaningful.
- The models see only 10 transaction-level features and no links between accounts, which limits how well they can separate false alarms from real laundering.
- Pattern recall covers only laundering that belongs to a named pattern.
- The 1% budget ranks the whole test period at once, not day by day.
