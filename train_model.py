import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ----------------------------------
# 1. Create project folder paths
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ----------------------------------
# 2. Generate demonstration data
# ----------------------------------

np.random.seed(42)

number_of_samples = 1000

karat = np.random.choice(
    [14, 18, 22, 24],
    size=number_of_samples
)

# Expected purity based on the selected karat
expected_purity = karat / 24 * 100

# Generate artificial purity readings around expected purity
purity = expected_purity + np.random.normal(
    loc=0,
    scale=1.5,
    size=number_of_samples
)

# Generate artificial density measurements
density = np.where(
    karat == 24,
    np.random.normal(19.3, 0.8, number_of_samples),
    np.where(
        karat == 22,
        np.random.normal(17.7, 0.8, number_of_samples),
        np.where(
            karat == 18,
            np.random.normal(15.5, 0.8, number_of_samples),
            np.random.normal(13.5, 0.8, number_of_samples)
        )
    )
)

# Generate an artificial hallmark indicator
hallmark = np.random.choice(
    [0, 1],
    size=number_of_samples
)

# Artificial labels based on invented rules.
# These labels are for demonstration only.
genuine = (
    (np.abs(purity - expected_purity) <= 2.5)
    & (density >= 12)
    & (density <= 21)
).astype(int)


# ----------------------------------
# 3. Create a DataFrame
# ----------------------------------

data = pd.DataFrame({
    "karat": karat,
    "purity": purity,
    "density": density,
    "hallmark": hallmark,
    "genuine": genuine
})

# Save the generated dataset
csv_path = DATASET_DIR / "gold_training.csv"
data.to_csv(csv_path, index=False)

print("Demonstration dataset created!")
print("Total samples:", len(data))
print("Dataset saved to:", csv_path)

print("\nFirst five rows:")
print(data.head())


# ----------------------------------
# 4. Select features and target
# ----------------------------------

X = data[["karat", "purity", "density", "hallmark"]]

y = data["genuine"]


# ----------------------------------
# 5. Split training and test data
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------------
# 6. Create the ML model
# ----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ----------------------------------
# 7. Train the model
# ----------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")


# ----------------------------------
# 8. Evaluate the model
# ----------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Fake (demonstration)", "Genuine (demonstration)"],
        zero_division=0
    )
)


# ----------------------------------
# 9. Save the trained model
# ----------------------------------

model_path = MODEL_DIR / "gold_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Model location:", model_path)

print(
    "\nWARNING: This model uses synthetic data and "
    "cannot authenticate real gold."
)

