import csv
from pathlib import Path

RAW_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering"
]

CLEAN_COLUMNS = [
    "timestamp", "sender_bank", "sender_account",
    "receiver_bank", "receiver_account", "amount_received",
    "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering"
]


def load_transactions(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        if next(reader) != RAW_COLUMNS:
            raise ValueError("Unexpected CSV column names")

        for row_number, values in enumerate(reader, start=1):
            if len(values) != len(CLEAN_COLUMNS):
                raise ValueError(f"Invalid number of columns in row {row_number}")

            transaction = dict(zip(CLEAN_COLUMNS, values))
            transaction["transaction_id"] = row_number

            yield transaction


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "HI-Small_Trans.csv"

    label_counts = {"0": 0, "1": 0}
    missing_counts = {column: 0 for column in CLEAN_COLUMNS}
    for transaction in load_transactions(data_path):
        label = transaction["is_laundering"]

        if label not in label_counts:
            raise ValueError(f"Unexpected label at row {transaction['transaction_id']}")

        label_counts[label] += 1
        for column in CLEAN_COLUMNS:
            if not transaction[column].strip():
                missing_counts[column] += 1
    print("Total transactions:", sum(label_counts.values()))
    print("Legitimate transactions:", label_counts["0"])
    print("Laundering transactions:", label_counts["1"])
    print("Missing values per column:", missing_counts)


    