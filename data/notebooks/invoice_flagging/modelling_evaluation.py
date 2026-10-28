
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def train_random_forest(x_train, y_train):
    """Tune and train a Random Forest classifier."""

    class_counts = y_train.value_counts()

    if y_train.nunique() < 2:
        raise ValueError(
            "The training data contains only one class. "
            "A binary classifier needs both classes."
        )

    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        raise ValueError(
            "The training data needs at least two records per class "
            "to perform cross-validation."
        )

    cv_splits = min(5, min_class_count)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=1
                ),
            ),
        ]
    )

    # Eight parameter combinations instead of 216.
    # With five-fold CV, that is up to 40 model fits.
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10],
        "classifier__min_samples_leaf": [1, 2],
    }

    scorer = make_scorer(
        f1_score,
        pos_label=1,
        zero_division=0
    )

    cross_validation = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cross_validation,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("\nStarting Random Forest training...", flush=True)

    grid_search.fit(x_train, y_train)

    print("\nTraining completed.", flush=True)
    print("Best parameters:", grid_search.best_params_)
    print(
        "Best cross-validation F1-score:",
        f"{grid_search.best_score_:.4f}"
    )

    return grid_search


def evaluate_classifier(model, x_test, y_test, model_name):
    """Evaluate the fitted classifier on the held-out test set."""

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not flagged", "Flagged"],
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(f"\n{'=' * 45}")
    print(f"{model_name} — Test Performance")
    print(f"{'=' * 45}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score for flagged invoices: {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print("Rows = actual [Not flagged, Flagged]")
    print("Columns = predicted [Not flagged, Flagged]")
    print(matrix)

    return {
        "accuracy": accuracy,
        "f1_score_flagged": f1,
        "confusion_matrix": matrix,
    }
