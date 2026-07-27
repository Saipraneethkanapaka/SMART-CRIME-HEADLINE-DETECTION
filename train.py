import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# Remove Missing Values
# ==========================================

df = df.dropna(
    subset=["title", "is_crime_report"]
)


print("\nClass Distribution:")
print(df["is_crime_report"].value_counts())


# ==========================================
# Features and Labels
# ==========================================

X = df["title"].astype(str)

y = df["is_crime_report"]


# ==========================================
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1,2),
    max_features=20000
)


X = vectorizer.fit_transform(X)


print("\nTF-IDF Features:", X.shape)


# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# Models
# ==========================================

models = {


    "Logistic Regression":

    LogisticRegression(
        C=2,
        max_iter=3000,
        random_state=42
    ),



    "Naive Bayes":

    MultinomialNB(),



    "SVM":

    SVC(
        kernel="linear",
        probability=True,
        random_state=42
    ),



    "Random Forest":

    RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),



    "XGBoost":

    XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

}



# ==========================================
# Train and Compare Models
# ==========================================

best_model = None
best_accuracy = 0
best_name = ""


for name, model in models.items():

    print("\n================================")
    print(name)
    print("================================")


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


    print(
        classification_report(
            y_test,
            prediction
        )
    )


    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_name = name



# ==========================================
# Save Best Model
# ==========================================

joblib.dump(
    best_model,
    "crime_headline_model.pkl"
)


joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)



print("\n================================")
print("BEST MODEL :", best_name)
print(
    "BEST ACCURACY :",
    round(best_accuracy * 100, 2),
    "%"
)

print("Model Saved Successfully!")
print("================================")