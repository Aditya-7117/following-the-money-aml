import csv
from pathlib import Path

file_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "HI-Small_Trans.csv"

with file_path.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)

    print("Column names:", next(reader))
    print("First transaction:", next(reader))