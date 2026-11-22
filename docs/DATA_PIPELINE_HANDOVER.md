# Data pipeline handover | CSC1181 Group 18

**Contributor:** Aditya Magar (data preparation / Person 1)  
**Repository:** `following-the-money-aml`  
**Document prepared:** 22 November 2026  
**Scope:** Data loading, validation, chronological feature generation, split assignment, reproducibility checks, and the outputs handed to EDA, modelling, and evaluation. This is an **engineering handover**, not the group's final academic report.

## 1. What was completed

The source pipeline prepares IBM AMLworld **HI-Small** transaction records for the team's statistical analysis and machine learning work. It:

1. Reads the original transaction CSV and rejects invalid headers, malformed rows, invalid labels, and invalid amounts.
2. Preserves the original CSV row number as `transaction_id` and sorts transactions chronologically (breaking timestamp ties by that ID).
3. Generates **10 model features**, including rolling transaction counts and distinct counterpart counts over the *preceding* 1 and 24 hours.
4. Assigns each transaction to `train`, `val`, `test_main`, or `test_tail` using the provisional cutoffs in `config/data.json`.
5. Writes separate **feature**, **label**, and **split-assignment** CSVs, joined by `transaction_id`, plus a run manifest and a check summary.
6. Includes automated tests for correctness, historical-window rules, invalid inputs, output alignment, and reproducibility.

Last recorded full-run checks in [`artifacts/data_checks.json`](../artifacts/data_checks.json): **5,078,345 records**, **5,177 positive laundering labels**, **0 duplicate transaction IDs**. The author reported **101 passing automated tests** on 9 November 2026. Each teammate should run the tests locally rather than assuming this status applies to their machine.

## 2. How the pieces connect

```text
IBM HI-Small_Trans.csv (download separately)
         |
         v
src/data/load.py          CSV headers, types/values, stable transaction_id
         |
         v
scripts/prepare_data.py   sample selection OR bounded-memory full chronological sort
         |
         v
src/data/features.py      10 features, using only permitted earlier activity
         |
         +--------------------------+
         |                          |
         v                          v
src/data/split.py          label kept separate
(config/data.json)         (is_laundering)
         |                          |
         +-------------+------------+
                       v
        data/processed/{sample,full}_features.csv
        data/processed/{sample,full}_labels.csv
        data/processed/{sample,full}_split_assignments.csv
                       |
                       v
        data/processed/manifest.json   (--full)
        artifacts/data_checks.json      (--full)
```

**Important:** The raw dataset and generated `data/processed/` CSVs are intentionally **not tracked in Git**. GitHub contains the code, documentation, configuration, tests, and the recorded small check summary, not the 5-million-row outputs. Each member needs the correct raw CSV to reproduce them.

## 3. Which file does what?

| File | Role | What to look at in the code |
|---|---|---|
| [`src/data/load.py`](../src/data/load.py) | Reads raw CSV records, standardises column names, assigns row IDs, rejects malformed rows or invalid labels/amounts. | `RAW_COLUMNS`, `CLEAN_COLUMNS`, `load_transactions()` |
| [`src/data/features.py`](../src/data/features.py) | Creates 10 model-ready feature values per transaction from the current record and strictly prior history. | `FEATURE_COLUMNS`, `_AccountHistory`, `generate_features()` |
| [`src/data/split.py`](../src/data/split.py) | Reads/validates configuration and assigns a chronological partition. | `load_config()`, `get_split_boundaries()`, `assign_split()` |
| [`config/data.json`](../config/data.json) | Holds the **provisional** split boundaries and documented feature/window contract. | `split_boundaries`, `feature_columns`, `rolling_window_rules` |
| [`scripts/prepare_data.py`](../scripts/prepare_data.py) | Command-line orchestrator: sample or full processing, sorting, writing three outputs, validations, hashes and manifest. | `main()`, `prepare_sample()`, `prepare_full()`, `sort_transactions_external()`, `stream_and_write_prepared_data()` |
| [`docs/data_dictionary.csv`](data_dictionary.csv) | Detailed description of every identifier, context column, feature, target and split column. | `model_feature` column |
| [`docs/data_dictionary.md`](data_dictionary.md) | Human-readable mapping between original and standardised raw column names. | Sender vs receiver account naming |
| [`docs/data_source.md`](data_source.md) | Dataset source, licence, original file hashes, dates and initial data checks. | Raw input SHA-256 |
| [`artifacts/data_checks.json`](../artifacts/data_checks.json) | Recorded counts, split summary and original/output file hashes from the author's full run. | `row_counts`, `split_counts`, `output_files` |
| `tests/test_load.py` | Invalid-input and basic loader tests. | `TestLoader` |
| `tests/data/test_features.py` | Historical-window, no-future-leakage, same-minute and configuration tests. | Test classes and edge cases |
| `tests/data/test_split.py` | Boundary/invalid-configuration tests. | Exact cutoff tests |
| `tests/data/test_prepare_data.py` | Sample and full orchestration tests on synthetic inputs. | End-to-end and alignment tests |

## 4. Reproduce on your own laptop

Run commands **from the repository's root folder**. The pipeline was developed with Python **3.12** and uses Python's standard library. The full preparation script imports the Unix `resource` module, so these exact commands are intended for **macOS/Linux**, not unmodified native Windows Python.

### A. Get the code and dataset

```bash
git clone https://github.com/Aditya-7117/following-the-money-aml.git
cd following-the-money-aml
python3.12 --version
```

If you already cloned it, keep your changes safe on your own branch and **fetch the updated main** instead of cloning again. The exact merge/rebase workflow depends on your uncommitted work.

Download the **IBM Transactions for Anti-Money Laundering (AML)** HI-Small files from the [dataset source](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). Place them at:

```text
data/raw/HI-Small_Trans.csv
data/raw/HI-Small_Patterns.txt
```

The `.txt` file is a source artifact **not used as a model feature**. The `.csv` file is needed to run the pipeline. Do not commit either file. The expected `HI-Small_Trans.csv` SHA-256 is recorded in [`docs/data_source.md`](data_source.md).

### B. Run the test suite first

```bash
python3.12 -m unittest discover -s . -p 'test_*.py'
```

All tests are designed to run from the checked-in code; the synthetic unit tests do not need a 5-million-row data file. The previous agent run reported **101 tests passing**, but verify the result locally.

### C. Fast sample check

```bash
python3.12 scripts/prepare_data.py --sample
```

This selects the **earliest 100,000 transactions**, not a random or representative sample, and writes:

```text
data/processed/sample_features.csv
data/processed/sample_labels.csv
data/processed/sample_split_assignments.csv
```

In the author's recorded sample run, all 100,000 records were in `train`, with **7 laundering labels**. This sample is for checking the interface and runtime, **not** for unbiased model evaluation.

### D. Full processing

```bash
python3.12 scripts/prepare_data.py --full
```

The full run sorts transactions using bounded-memory chunks, produces the complete files below, computes SHA-256 hashes and creates a manifest. Allow adequate free disk space for the raw data, temporary sort chunks and final outputs.

```text
data/processed/full_features.csv
data/processed/full_labels.csv
data/processed/full_split_assignments.csv
data/processed/manifest.json
artifacts/data_checks.json
```

The last recorded run completed in **86.97 seconds** with **955.70 MB peak reported memory** on the author's environment; speed and memory will differ by machine. The recorded reference hashes for each full output are in [`artifacts/data_checks.json`](../artifacts/data_checks.json). Check your run's `data/processed/manifest.json` against those hashes if you used exactly the same dataset, code, and configuration. Execution timestamps and runtime fields naturally differ.

For available command-line options, run:

```bash
python3.12 scripts/prepare_data.py --help
```

Examples of supported options include `--data-path`, `--output-dir`, `--sample-size`, `--chunk-size`, `--artifacts-dir`, and `--config-path`.

## 5. How to use the outputs correctly

The three full CSV files share **`transaction_id`**. Use this as the join key:

| Output | Columns | Used for |
|---|---|---|
| `full_features.csv` | `transaction_id` + exactly **10 features** | Model input `X` (excluding ID) |
| `full_labels.csv` | `transaction_id`, `is_laundering` | Ground-truth target `y` |
| `full_split_assignments.csv` | `transaction_id`, `split` | Select train/validation/test partitions |

**Modelling rules:** Never include `transaction_id`, `is_laundering`, or `split` as predictors. Fit categorical encoders and other trainable preprocessing **on the training subset only**, then apply the fitted transforms to validation and test. The model decides predictions using feature columns; the label is the known answer used for training or evaluation.

**Yashasvi (EDA):** The raw CSV and dictionaries are available to explore the dataset. For plots derived from the model feature set, use `full_features.csv` joined to labels/splits by `transaction_id`, and state which partition(s) your plot covers.

**Sofia (models):** Use `train` to fit, `val` for model/threshold selection, and hold out test partitions for final evaluation. Check the exact 10-column schema and outputs on your own machine.

**Priyanka (evaluation):** Report `test_main` separately and a **full test view equal to `test_main` + `test_tail`**. These are two evaluation views of one test period, not unrelated test sets.

## 6. Feature design decisions and correctness

- **Current-transaction features:** `log_amount_paid`, `payment_currency`, `payment_format`, `same_bank`.
- **Prior-activity features (six):** sender and receiver transaction counts over 1h/24h and their unique counterpart counts over 24h.
- **Account identity:** `(bank_id, account_id)` tuple, not account number alone. This avoids mixing equal account numbers from different banks.
- **Windows:** for a transaction at `t`, earlier activity counts only if `t - window < earlier_timestamp < t`. Both boundaries are **exclusive**.
- **Same-minute protection:** transactions at the same timestamp are all scored before that minute updates the account histories, so they cannot appear in one another's features.
- **Out-of-order input:** the feature generator rejects decreasing timestamps; the preparation script sorts input by timestamp then original transaction ID.
- **Bounded memory:** the full pipeline uses external chunk sorting and a rolling 24-hour history rather than loading every historical record into active feature state.
- **No label leakage into features:** the laundering label, account IDs and raw timestamps are not included in the 10 model feature columns.

**Configuration note:** The JSON records 1-hour/24-hour window values, but `features.py` currently defines these Python intervals explicitly. Consistency is guarded by tests. Changing the JSON alone **will not change the feature algorithm**. Split cutoff dates, by contrast, are read from JSON by the split/preparation code.

## 7. Provisional chronological splits and unusual late tail

These split cutoffs are still marked **provisional awaiting team approval** in `config/data.json`:

| Partition | Time range | Transactions | Laundering |
|---|---|---:|---:|
| `train` | Before 2022/09/06 13:36 | 3,046,861 | 2,297 |
| `val` | 2022/09/06 13:36 to before 2022/09/08 16:12 | 1,015,602 | 1,082 |
| `test_main` | 2022/09/08 16:12 to before 2022/09/11 00:00 | 1,014,774 | 1,143 |
| `test_tail` | From 2022/09/11 00:00 onwards | 1,108 | 655 |

The tail has **655 / 1,108 = 59.12%** laundering labels, while the main test has **1,143 / 1,014,774 ≈ 0.113%**. The raw-data check summary describes all 1,108 tail transactions as ACH. This is a dataset distribution anomaly, **not proof of corruption**. The agreed evaluation plan is to compare `test_main` against its union with `test_tail` rather than silently discard the anomaly.

## 8. Troubleshooting and handover checklist

| Symptom | Check |
|---|---|
| `FileNotFoundError` for raw CSV | Put `HI-Small_Trans.csv` inside `data/raw/` or use `--data-path`. |
| `python3.12` not found | Install/select Python 3.12; verify the interpreter used to run commands. |
| `ModuleNotFoundError: resource` | Native Windows Python lacks the Unix module. Run on macOS/Linux, or adapt/test platform handling separately. |
| No `data/processed/` on GitHub | Expected: generated datasets are Git-ignored. Run `--sample` or `--full` locally. |
| Feature/label mismatch | Re-run preparation, compare `transaction_id` and verify `manifest.json` hashes. |
| Different split counts | Confirm that everyone uses the same raw CSV hash and `config/data.json` cutoffs. |
| Changed feature results | Check code revision, input SHA-256, window rules, and the same-minute history logic. |

**Before declaring the handover complete:**

- [ ] Teammates can see this document from the root `README.md`.
- [ ] Yashasvi confirms he can locate the input data, run commands and understand the file responsibilities.
- [ ] Sofia runs tests and `--sample` on her laptop, confirms the model interface, and if feasible reproduces `--full` and its output hashes.
- [ ] All members approve the split-boundary policy before final metrics are frozen.
- [ ] Priyanka understands main-test versus full-test evaluation and keeps test outcomes separate from model-selection decisions.
- [ ] Any corrections go through reviewed Git commits/PRs rather than informal edits on `main`.

**Limits of this handover:** It documents the data preparation implementation checked on the `main` branch as of 22 November 2026. It does **not** assert that model training, EDA, evaluation, or other members' local reproduction are already complete. Those require separate evidence and their own documentation.
