import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ===========================================================================
# Adding more sample models to the model.py file as below :

def model1():
    print("Model 1: Logistic Regression")

def model2():
    print("Model 2: Random Forest")

def model3():
    print("Model 3: Gradient Boosting")

# ======================================================================


def load_sample_data():
    data = {
        "amount": [50, 120, 3000, 80, 250, 7000, 60, 90, 1500, 110, 40, 2500],
        "merchant": [
            "grocery", "online", "travel", "grocery", "online", "travel",
            "grocery", "food", "online", "food", "grocery", "travel"
        ],
        "hour": [9, 15, 23, 10, 2, 1, 14, 11, 21, 7, 17, 3],
        "country": ["US", "US", "CA", "US", "GB", "US", "CA", "US", "US", "GB", "US", "CA"],
        "is_fraud": [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)


def train_fraud_model():
    df = load_sample_data()

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    numeric_features = ["amount", "hour"]
    categorical_features = ["merchant", "country"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nROC AUC:", roc_auc_score(y_test, probabilities))

    return model


if __name__ == "__main__":
    train_fraud_model()
