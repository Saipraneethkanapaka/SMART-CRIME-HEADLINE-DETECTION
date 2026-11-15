
import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATASET_PATH = "dataset.csv"

MODEL_PATH = "crime_headline_model.pkl"
VECTORIZER_PATH = "count_vectorizer.pkl"
METRICS_PATH = "model_metrics.pkl"
MODEL_NAME_PATH = "best_model_name.pkl"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"{DATASET_PATH} not found. "
        "Place dataset.csv in the same folder as train.py."
    )

df = pd.read_csv(DATASET_PATH)

required_columns = ["title", "is_crime_report"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from dataset.csv"
        )

df = df[required_columns].copy()

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

print("\nCleaning dataset...")

df.dropna(subset=["title", "is_crime_report"], inplace=True)

df["title"] = df["title"].astype(str).str.strip()

df = df[df["title"] != ""]

# Normalize labels
def convert_label(value):

    value = str(value).strip().lower()

    crime_labels = {
        "1", "1.0", "crime", "yes", "true"
    }

    non_crime_labels = {
        "0", "0.0", "not crime", "non-crime",
        "non crime", "not_crime", "no", "false"
    }

    if value in crime_labels:
        return 1

    if value in non_crime_labels:
        return 0

    raise ValueError(
        f"Unknown label found in dataset: {value}"
    )


df["is_crime_report"] = df["is_crime_report"].apply(convert_label)

# Remove exact duplicate title-label combinations
df.drop_duplicates(
    subset=["title", "is_crime_report"],
    inplace=True
)

# Remove titles that have conflicting labels
label_counts = df.groupby("title")["is_crime_report"].nunique()

conflicting_titles = label_counts[label_counts > 1].index

df = df[~df["title"].isin(conflicting_titles)]

df.reset_index(drop=True, inplace=True)

print("Cleaned dataset shape:", df.shape)

print("\nClass distribution:")
print(df["is_crime_report"].value_counts())

if df["is_crime_report"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both classes: Crime (1) and Non-Crime (0)."
    )

if df["is_crime_report"].value_counts().min() < 5:
    raise ValueError(
        "Not enough samples in one of the classes for stratified splitting."
    )


# --------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df["title"]
y = df["is_crime_report"]


# --------------------------------------------------
# 5. SPLIT DATASET
# 60% TRAINING
# 20% VALIDATION
# 20% TESTING
# --------------------------------------------------

print("\nSplitting dataset...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. TEXT VECTORIZATION
# --------------------------------------------------

print("\nConverting headlines into numerical features...")

vectorizer = CountVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words=None
)

# Fit only on training data to avoid test-data leakage
X_train_vectorized = vectorizer.fit_transform(X_train)

X_val_vectorized = vectorizer.transform(X_val)

X_test_vectorized = vectorizer.transform(X_test)

print("Vocabulary size:", len(vectorizer.vocabulary_))


# --------------------------------------------------
# 7. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=1.0
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        algorithm="brute",
        n_jobs=-1
    )
}


# --------------------------------------------------
# 8. TRAIN MODELS AND EVALUATE ON VALIDATION DATA
# --------------------------------------------------

print("\nTraining all models...")

validation_results = {}

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)

    try:

        model.fit(
            X_train_vectorized,
            y_train
        )

        predictions = model.predict(
            X_val_vectorized
        )

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        validation_results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        trained_models[model_name] = model

        print("Accuracy :", round(accuracy * 100, 2), "%")
        print("Precision:", round(precision * 100, 2), "%")
        print("Crime Recall:", round(recall * 100, 2), "%")
        print("F1 Score :", round(f1 * 100, 2), "%")

    except Exception as e:

        print("Error training", model_name, ":", e)


if not validation_results:
    raise RuntimeError("No model was successfully trained.")


# --------------------------------------------------
# 9. SELECT BEST MODEL
# PRIORITY: CRIME RECALL
# TIE-BREAKER: F1 SCORE
# --------------------------------------------------

print("\nSelecting the best model...")

best_model_name = max(
    validation_results,
    key=lambda name: (
        validation_results[name]["Recall"],
        validation_results[name]["F1 Score"]
    )
)

print("\nSelected Model:", best_model_name)

print(
    "Validation Crime Recall:",
    round(
        validation_results[best_model_name]["Recall"] * 100,
        2
    ),
    "%"
)

print(
    "Validation F1 Score:",
    round(
        validation_results[best_model_name]["F1 Score"] * 100,
        2
    ),
    "%"
)


# --------------------------------------------------
# 10. COMBINE TRAINING AND VALIDATION DATA
# --------------------------------------------------

print("\nRetraining models using training + validation data...")

X_train_final = pd.concat(
    [X_train, X_val]
)

y_train_final = pd.concat(
    [y_train, y_val]
)

X_train_final_vectorized = vectorizer.transform(
    X_train_final
)


# --------------------------------------------------
# 11. TRAIN ALL MODELS AGAIN
# AND TEST ON UNSEEN TEST DATA
# --------------------------------------------------

test_results = {}

final_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Final training:", model_name)

    try:

        model.fit(
            X_train_final_vectorized,
            y_train_final
        )

        predictions = model.predict(
            X_test_vectorized
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        test_results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        final_models[model_name] = model

        print("Accuracy :", round(accuracy * 100, 2), "%")
        print("Precision:", round(precision * 100, 2), "%")
        print("Crime Recall:", round(recall * 100, 2), "%")
        print("F1 Score :", round(f1 * 100, 2), "%")

    except Exception as e:

        print("Error during final training:", e)


# --------------------------------------------------
# 12. GET SELECTED MODEL
# --------------------------------------------------

if best_model_name not in final_models:
    raise RuntimeError(
        "The selected model could not be trained during final training."
    )

best_model = final_models[best_model_name]


# --------------------------------------------------
# 13. FINAL MODEL EVALUATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

y_pred = best_model.predict(
    X_test_vectorized
)

print("\nSelected Model:", best_model_name)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Crime", "Crime"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Explanation:")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print(
    "\nFalse Negatives are important because "
    "they represent crime headlines predicted as Non-Crime."
)


# --------------------------------------------------
# 14. DISPLAY MODEL COMPARISON
# --------------------------------------------------

print("\n" + "=" * 60)
print("ALL MODEL TEST RESULTS")
print("=" * 60)

results_df = pd.DataFrame(test_results).T

results_df = results_df.sort_values(
    by=["Recall", "F1 Score"],
    ascending=False
)

print(
    (results_df * 100).round(2).to_string()
)


# --------------------------------------------------
# 15. SAVE MODEL AND OTHER FILES
# --------------------------------------------------

print("\nSaving model files...")

joblib.dump(
    best_model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

joblib.dump(
    test_results,
    METRICS_PATH
)

joblib.dump(
    best_model_name,
    MODEL_NAME_PATH
)

# Save comparison results as CSV
results_df.to_csv(
    "model_comparison.csv"
)


# --------------------------------------------------
# 16. SAVE CONFUSION MATRIX
# --------------------------------------------------

confusion_df = pd.DataFrame(
    cm,
    index=["Actual Non-Crime", "Actual Crime"],
    columns=["Predicted Non-Crime", "Predicted Crime"]
)

confusion_df.to_csv(
    "confusion_matrix.csv"
)


# --------------------------------------------------
# 17. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nBest Model:", best_model_name)

print("\nFiles Saved:")

print("1.", MODEL_PATH)
print("2.", VECTORIZER_PATH)
print("3.", METRICS_PATH)
print("4.", MODEL_NAME_PATH)
print("5. model_comparison.csv")
print("6. confusion_matrix.csv")

print("\nYou can now run your Streamlit application.")
