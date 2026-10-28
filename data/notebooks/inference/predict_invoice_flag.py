
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "invoice_flagging"
    / "models"
    / "predict_flag_invoice.pkl"
)
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_model():
    """Load the trained invoice-flagging pipeline."""
    return joblib.load(MODEL_PATH)


def predict_invoice_flag(input_data):
    """Return the invoice data with its predicted flag."""
    input_df = pd.DataFrame(input_data)

    missing = [column for column in FEATURES if column not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    input_df = input_df[FEATURES].copy()
    model = load_model()

    input_df["Predicted Flag"] = model.predict(input_df)
    input_df["Flag Status"] = input_df["Predicted Flag"].map({
        0: "Not flagged",
        1: "Flagged",
    })

    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [10, 5],
        "invoice_dollars": [18500, 9000],
        "Freight": [250, 150],
        "total_item_quantity": [10, 5],
        "total_item_dollars": [18500, 8990],
    }

    print(predict_invoice_flag(sample_data))