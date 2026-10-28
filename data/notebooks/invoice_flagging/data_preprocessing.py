
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_invoice_data(db_path: str | Path) -> pd.DataFrame:
    """Load invoice and purchase information from SQLite."""

    db_path = Path(db_path)

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found: {db_path.resolve()}"
        )

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(
                julianday(p.ReceivingDate) - julianday(p.PODate)
            ) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        julianday(vi.InvoiceDate) - julianday(vi.PODate)
            AS days_po_to_invoice,
        julianday(vi.PayDate) - julianday(vi.InvoiceDate)
            AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber;
    """

    conn = sqlite3.connect(str(db_path))

    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError("The invoice query returned no records.")

    print(f"Loaded {len(df)} invoice records.", flush=True)

    return df


def create_invoice_risk_label(row) -> int:
    """
    Flag an invoice if:
    1. Invoice and purchase dollar totals differ by more than $5.
    2. Average receiving delay exceeds 10 days.
    """

    invoice_dollars = row["invoice_dollars"]
    total_item_dollars = row["total_item_dollars"]
    receiving_delay = row["avg_receiving_delay"]

    if (
        pd.notna(invoice_dollars)
        and pd.notna(total_item_dollars)
        and abs(invoice_dollars - total_item_dollars) > 5
    ):
        return 1

    if (
        pd.notna(receiving_delay)
        and receiving_delay > 10
    ):
        return 1

    return 0


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target column to the DataFrame."""

    required_columns = [
        "invoice_dollars",
        "total_item_dollars",
        "avg_receiving_delay",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns needed for labeling: {missing}"
        )

    df = df.copy()

    df["flag_invoice"] = df.apply(
        create_invoice_risk_label,
        axis=1
    ).astype(int)

    print(
        "Invoice label distribution:",
        df["flag_invoice"].value_counts().sort_index().to_dict(),
        flush=True
    )

    return df


def split_data(df, features, target):
    """Split the data into training and testing sets."""

    missing = [
        column for column in [*features, target]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns required for training: {missing}"
        )

    x = df[features].copy()
    y = df[target].copy()

    class_counts = y.value_counts()

    if y.nunique() < 2:
        raise ValueError(
            "Classification requires both classes, 0 and 1. "
            "Your current labeling rules produced only one class."
        )

    if class_counts.min() < 3:
        raise ValueError(
            "At least 3 records per class are required for this "
            "stratified train/test split. Check your label distribution."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {len(x_train)}", flush=True)
    print(f"Testing records: {len(x_test)}", flush=True)

    return x_train, x_test, y_train, y_test
