import pandas as pd
import joblib
from pathlib import Path

# Find and load the trained model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gold_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("-" * 40)

# Enter the gold details

karat = float(input("Enter karat (14, 18, 22, or 24): "))
purity = float(input("Enter purity percentage: "))
density = float(input("Enter density in g/cm3: "))
hallmark = int(input("Hallmark present? Enter 1 for Yes or 0 for No: "))

# Prepare the data for prediction

test_data = pd.DataFrame([{
"karat": karat,
"purity": purity,
"density": density,
"hallmark": hallmark
}])

# Predict the result

prediction = int(model.predict(test_data)[0])

probabilities = model.predict_proba(test_data)[0]
class_index = list(model.classes_).index(prediction)
confidence = probabilities[class_index] * 100

if prediction == 1:
    print("\nPrediction: Genuine (demonstration)")
else:
    print("\nPrediction: Fake (demonstration)")

print(f"Model-estimated class probability: {confidence:.2f}%")
print("\nNote: This model uses synthetic data and cannot authenticate real gold.")