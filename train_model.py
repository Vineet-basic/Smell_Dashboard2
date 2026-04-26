"""
train_model.py
──────────────
Replace the synthetic data below with your real sensor CSV dataset,
then run:  python train_model.py

The script will:
  1. Train a Random Forest classifier
  2. Save smell_classifier.pkl  →  model/
  3. Save label_encoder.pkl     →  model/

Sensor order expected by the Flask server:
  MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135
"""

import os, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Configuration ─────────────────────────────────────────────────────────────
SENSOR_NAMES = ["MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9", "MQ135"]
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load your dataset here ────────────────────────────────────────────────────
# Option A – CSV file
#   import pandas as pd
#   df = pd.read_csv("your_data.csv")
#   X  = df[SENSOR_NAMES].values
#   y  = df["label"].values

# Option B – Synthetic demo data (9 smell classes, 100 samples each)
SMELL_CLASSES = [
    "Smoke / LPG",
    "Alcohol Vapour",
    "Natural Gas",
    "LPG / Natural Gas",
    "LPG / Butane",
    "Carbon Monoxide",
    "Hydrogen Gas",
    "CO / Combustibles",
    "Air Quality / NH3",
]

rng = np.random.default_rng(42)
X_parts, y_parts = [], []

for class_idx, smell in enumerate(SMELL_CLASSES):
    # Each class has a dominant sensor (same index as class) with higher values
    samples = rng.integers(100, 400, size=(100, 9))
    # Elevate the dominant sensor reading
    samples[:, class_idx] = rng.integers(600, 1000, size=100)
    X_parts.append(samples)
    y_parts.extend([smell] * 100)

X = np.vstack(X_parts).astype(float)
y = np.array(y_parts)

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Train classifier ──────────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n── Classification Report ─────────────────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Feature importance ────────────────────────────────────────────────────────
print("── Sensor Importances ───────────────────────────────────────────────")
for s, imp in sorted(zip(SENSOR_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 60)
    print(f"  {s:>6}  {imp:.4f}  {bar}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "smell_classifier.pkl")
label_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

with open(model_path, "wb") as f:
    pickle.dump(clf, f)
with open(label_path, "wb") as f:
    pickle.dump(le, f)

print(f"\n✅ Model saved  →  {model_path}")
print(f"✅ Encoder saved →  {label_path}")
print("\nRestart the Flask server (or POST to /api/reload_model) to use the model.")
