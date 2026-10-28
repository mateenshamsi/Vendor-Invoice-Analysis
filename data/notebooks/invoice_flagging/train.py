
from pathlib import Path
import joblib

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
)

from modelling_evaluation import (
    train_random_forest,
    evaluate_classifier,
)


BASE_DIR = Path(__file__).resolve().parent

# Update this if your SQLite database is stored elsewhere.
DB_PATH = Path(r"C:\Users\My-PC\Pytohn\data\inventory.db")

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "predict_flag_invoice.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"


def main():
    print("Invoice classification pipeline started.", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the data.
    df = load_invoice_data(DB_PATH)

    # 2. Generate the target labels.
    df = apply_labels(df)

    # 3. Create training and testing sets.
    x_train, x_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # 4. Train and tune the model.
    grid_search = train_random_forest(
        x_train,
        y_train
    )

    # GridSearchCV refits the winning pipeline on all training data.
    best_model = grid_search.best_estimator_

    # 5. Evaluate on the held-out test set.
    evaluate_classifier(
        best_model,
        x_test,
        y_test,
        "Random Forest Classifier"
    )

    # 6. Save the entire fitted pipeline, including the imputer.
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nModel saved successfully: {MODEL_PATH}", flush=True)
    print("Pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
