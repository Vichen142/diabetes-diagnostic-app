import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. Load the cleaned dataset
df = pd.read_csv("diabetes_cleaned.csv")

# 2. Explicitly select only the 8 clinical input features
feature_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

X = df[feature_cols]
y = df["Outcome"]

# 3. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Production ML Pipeline
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=6,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

# 5. Fit the model
pipeline.fit(X_train, y_train)

# 6. Evaluation metrics
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("=== Clinical Evaluation Report ===")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Diabetic"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Save pipeline artifact
joblib.dump(pipeline, "diabetes_pipeline.joblib")
print("\n[SUCCESS] diabetes_pipeline.joblib generated successfully!")