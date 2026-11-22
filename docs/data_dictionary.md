# Data Dictionary

Dataset: HI-Small_Trans.csv

Each row represents one transaction.

## Column Definitions

| Original Column | Our Column Name | Meaning |
|---|---|---|
| Timestamp | timestamp | Date and time of the transaction |
| From Bank | sender_bank | Sending bank identifier |
| Account (first) | sender_account | Sending account identifier |
| To Bank | receiver_bank | Receiving bank identifier |
| Account (second) | receiver_account | Receiving account identifier |
| Amount Received | amount_received | Amount received by the recipient |
| Receiving Currency | receiving_currency | Currency of the received amount |
| Amount Paid | amount_paid | Amount sent by the sender |
| Payment Currency | payment_currency | Currency of the paid amount |
| Payment Format | payment_format | Transaction payment method |
| Is Laundering | is_laundering | Target label: 0 = legitimate, 1 = laundering |

## Additional Column

- transaction_id: Original CSV row number, starting from 1.

## Important Notes

- Bank and account identifiers are kept as strings to preserve leading zeros.
- The original CSV contains two columns named Account.
- Our loader gives them separate names to avoid confusion.
- The laundering label is kept separate from the model's input features.

## Initial Data Checks

- Total transactions: 5,078,345
- Legitimate transactions: 5,073,168
- Laundering transactions: 5,177
- Missing or blank values: 0 across all 11 original columns