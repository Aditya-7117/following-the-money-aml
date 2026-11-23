"""
Read HI-Small_Patterns.txt and find the transaction_id of each record.

The patterns file is used ONLY to check results after the models are done.
It is never used to build features or to train a model.
"""

import pandas as pd

# The same 11 column names that Aditya's loader uses
from src.data.load import CLEAN_COLUMNS


def read_patterns(path):
    """Turn the patterns file into a table with one row per record.

    The file is made of blocks like this:
        BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
        2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,...,ACH,1
        END LAUNDERING ATTEMPT - FAN-OUT
    """
    rows = []
    attempt_id = 0
    pattern_type = None

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt_id += 1
                # "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max ..." -> "FAN-OUT"
                pattern_type = line.split(" - ")[1].split(":")[0].strip()

            elif line.startswith("END LAUNDERING ATTEMPT"):
                pattern_type = None

            else:
                # a transaction record: the same 11 fields as the main CSV
                fields = line.split(",")
                if pattern_type is None or len(fields) != 11:
                    raise ValueError(f"Unexpected line in patterns file: {line}")
                rows.append([attempt_id, pattern_type] + fields)

    return pd.DataFrame(rows, columns=["attempt_id", "pattern_type"] + CLEAN_COLUMNS)


def match_patterns(patterns, csv_path, chunk_size=500_000):
    """Find each pattern record in the transactions CSV.

    transaction_id is the row number in the raw CSV (1, 2, 3, ...),
    the same rule Aditya's loader uses, so it joins to his output files.

    We compare all 11 columns as text, so "100.00" only matches "100.00".
    Each record gets a status:
      matched   - exactly one CSV row is the same
      unmatched - no CSV row is the same
      ambiguous - several CSV rows are the same (we do NOT pick one)

    Returns the patterns table with status and transaction_id added,
    and the total number of laundering rows in the CSV.
    """
    found = []
    n_laundering = 0
    unique_records = patterns[CLEAN_COLUMNS].drop_duplicates()

    # Read the big CSV in pieces so it fits in memory.
    # dtype=str keeps bank IDs like "021" and amounts exactly as written.
    reader = pd.read_csv(csv_path, dtype=str, header=0, names=CLEAN_COLUMNS,
                         keep_default_na=False, encoding="utf-8-sig",
                         chunksize=chunk_size)
    for chunk in reader:
        # the row index keeps counting across pieces, so +1 gives the row number
        chunk["transaction_id"] = chunk.index + 1
        n_laundering += int((chunk["is_laundering"] == "1").sum())
        found.append(chunk.merge(unique_records, on=CLEAN_COLUMNS))

    matches = pd.concat(found)

    # how many CSV rows matched each record, and the first matching ID
    counts = (matches.groupby(CLEAN_COLUMNS)["transaction_id"]
              .agg(n_matches="count", first_id="min")
              .reset_index())
    result = patterns.merge(counts, on=CLEAN_COLUMNS, how="left")
    result["n_matches"] = result["n_matches"].fillna(0).astype(int)

    # a record listed twice in the patterns file is also ambiguous
    listed_twice = result.duplicated(CLEAN_COLUMNS, keep=False)

    result["status"] = "matched"
    result.loc[result["n_matches"] == 0, "status"] = "unmatched"
    result.loc[(result["n_matches"] > 1) | listed_twice, "status"] = "ambiguous"

    # only a clean one-to-one match gets a transaction_id
    result["transaction_id"] = result["first_id"].where(result["status"] == "matched")
    result["transaction_id"] = result["transaction_id"].astype("Int64")

    return result.drop(columns="first_id"), n_laundering


def coverage_summary(result, n_laundering):
    """Counts for the report: how many records we could link, and how many not."""
    status = result["status"].value_counts()
    matched = result[result["status"] == "matched"]

    return {
        "attempts": int(result["attempt_id"].nunique()),
        "pattern_records": len(result),
        "matched": int(status.get("matched", 0)),
        "unmatched": int(status.get("unmatched", 0)),
        "ambiguous": int(status.get("ambiguous", 0)),
        "csv_laundering_rows": n_laundering,
        "laundering_with_pattern": len(matched),
        "laundering_without_pattern": n_laundering - len(matched),
        "matched_per_pattern": matched["pattern_type"].value_counts().to_dict(),
    }
