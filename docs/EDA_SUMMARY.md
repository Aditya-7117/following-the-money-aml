# Exploratory data analysis summary | CSC1181 Group 18

**Contributor:** Yasashwi (statistics and EDA)
**Branch:** `yasashwi/eda-profile`
**Data:** prepared HI-Small outputs from `scripts/prepare_data.py --full` (`data/processed/full_*`), joined on `transaction_id`. Raw timestamps are read from `data/raw/HI-Small_Trans.csv` only for the time analysis.
**Splits:** train only, unless a section says otherwise. All-split results are descriptive and are not used for feature or model choices.

## Files

| File | What it does |
|---|---|
| `src/analysis/data.py` | `load_eda_frame()` joins features, labels and splits (train only by default); `load_timestamps()` adds real datetimes |
| `src/analysis/stats.py` | `rate_table()` (laundering rate with 95% Wilson intervals and lift), `chi_square_test()` (with Cramér's V), `mann_whitney_test()` (with AUC and rank-biserial effect sizes) |
| `notebooks/yasashwi/01_data_profile.ipynb` | Size, class balance, data quality, basic distributions |
| `notebooks/yasashwi/02_formats_and_currencies.ipynb` | Payment format, currency, same-bank, and the ACH × currency interaction |
| `notebooks/yasashwi/03_amounts_and_activity.ipynb` | Amounts within currency and the six rolling-activity features |
| `notebooks/yasashwi/04_time_and_splits.ipynb` | Daily timeline, warm-up check, hour of day, test_tail and the split cutoffs |

To reproduce: run the data pipeline, then open each notebook with the project's Python environment and choose Run All.

## Key findings

1. **Laundering is extremely rare.** 5,177 of 5,078,345 transactions (0.10%); in train 0.075%, or 1 in 1,326. The data has no missing values or duplicate IDs. Accuracy is therefore not a useful metric, and effect sizes matter more than p-values: with 3 million rows almost every test is significant.

2. **ACH is the dominant signal.** ACH is 11% of train payments but 83% of train laundering. Its laundering rate is 0.56% (95% CI 0.54–0.59%), 7.5× the average (χ² = 12,026, df = 6, p < 0.001; Cramér's V = 0.063, which is small only because the outcome is so rare). The effect holds within every currency: ACH launders at roughly 23–65× the non-ACH rate. Wire and Reinvestment have no laundering at all in train.

3. **Currency matters little on its own** (Cramér's V = 0.010), with one interaction. Saudi Riyal payments are ACH as often as other currencies (10%), but Saudi Riyal ACH payments launder at 2.5%, against 0.35–0.68% for ACH in other currencies. Saudi Riyal non-ACH payments are below average (0.006%).

4. **Laundering crosses banks.** 98.4% of train laundering is between different banks; the cross-bank rate (0.093%) is about 15× the same-bank rate (0.006%).

5. **Amount has an inverted-U relationship.** Using each payment's percentile within its own currency, the laundering rate rises from 0.004% in the smallest decile to about 0.19% in deciles 8–9, then falls to 0.023% in the largest decile. Deciles 6–9 hold 81% of laundering. Within-currency amount is the strongest numeric signal (Mann–Whitney AUC = 0.67, rank-biserial 0.34).

6. **Sender activity has a U shape.** Senders with no outgoing payments in the previous 24 hours, and very busy senders (51+), both launder at about 0.145%, while moderately active senders are below average. Means and medians point in opposite directions (24h count: laundering median 1 vs 3, mean 1,452 vs 730), so rank-based tests and bucket charts were used.

7. **The "no sender history" effect is real, not a warm-up artefact.** After the first 24 hours of data, senders with no history launder at 0.34%, about 5.7× senders with history, and account for 43% of train laundering. The first day dilutes the effect, because many legitimate accounts also have empty histories then.

8. **Receivers paid by many different senders are high-risk.** Receivers with 6–10 distinct senders in the previous 24 hours launder at 0.75% (10× average). This resembles the fan-in pattern described by Altman et al. (2023), but the group is small (about 9,600 payments) and the interval is wide.

9. **Time.** The data has a weekly cycle (1 Sept 2022 was a Thursday): weekend volume is about 207k per day against about 480k on weekdays, while daily laundering counts stay at roughly 320–540, so the weekend rate is about double. Hour of day has a significant but weak effect (Cramér's V = 0.011), with most laundering in working hours. Time is not a model feature.

10. **test_tail is a simulation artefact.** Normal activity ends on 10 Sept. The 1,108 tail transactions (11–18 Sept) are all ACH and 59% laundering every day, with larger amounts, more same-bank payments and empty sender histories compared with ACH in test_main.

## What this means for the rest of the project

**Models (Sofia)**
- The amount and sender-activity effects are non-linear (inverted U and U). So is the ACH × Saudi Riyal effect: it depends on the two together. A decision tree captures these; logistic regression and Naive Bayes cannot without binned or interaction features. This is consistent with the tree having the best validation PR-AUC.
- The Bitcoin payment format and Bitcoin currency mark almost the same 87,143 rows, so the two features largely duplicate each other.
- No train laundering uses Wire or Reinvestment, so models are likely to treat those formats as safe.

**Evaluation (Priyanka)**
- Report `test_main` on its own and with `test_tail`, as planned. The tail's feature distributions differ strongly from everything else, so results on it will not reflect normal conditions.
- Test_main ends on a Saturday (10 Sept), when the laundering rate is highest (0.21%).

**Split cutoffs (team decision)**
- The provisional cutoffs give a chronological 60/20/20 split by rows. Val and test_main are similar (rates 0.107% and 0.113%; ACH share 12.1% and 13.6%; ACH rates 0.75% and 0.72%), which suits tuning on val and testing on test_main.
- Train's laundering rate is lower (0.075%), partly because it includes the high-volume, low-rate start of the data on 1–2 Sept. The weekday mix also differs: train contains a weekend, val none. Neither seems to require new cutoffs, but both should be stated in the report.

## Limitations

- The data is synthetic. Patterns such as "Wire never launders" reflect IBM's simulator, not real banking.
- Amounts are in 15 unconverted currencies, so amount comparisons use within-currency percentiles.
- Feature analysis uses train only. Patterns may shift slightly later in time; the laundering rate rises from train to val and test_main.
- Some high-risk groups (receiver fan-in, the Saudi Riyal) are small, so their exact rates are uncertain; the confidence intervals are shown in the notebooks.

## Reference

Altman, E. et al. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.16424.
