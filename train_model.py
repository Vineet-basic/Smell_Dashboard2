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
FEATURE_NAMES = [
    "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9", "MQ135",
    "MQ3_MQ4_Ratio", "MQ8_MQ3_Ratio", "MQ3_MQ135_Sum",
    "MQ5_MQ8_Ratio", "Total_Active_VOC",
]
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature Engineering (Synced with app.py) ──────────────────────────────────
def engineer_features_batch(X_raw: np.ndarray) -> np.ndarray:
    """
    X_raw: (N, 9) array in SENSOR_NAMES order.
    Returns: (N, 14) array in FEATURE_NAMES order.
    """
    # X_raw columns: 0:MQ2, 1:MQ3, 2:MQ4, 3:MQ5, 4:MQ6, 5:MQ7, 6:MQ8, 7:MQ9, 8:MQ135
    mq2   = X_raw[:, 0]
    mq3   = X_raw[:, 1]
    mq4   = X_raw[:, 2]
    mq5   = X_raw[:, 3]
    mq6   = X_raw[:, 4]
    mq7   = X_raw[:, 5]
    mq8   = X_raw[:, 6]
    mq9   = X_raw[:, 7]
    mq135 = X_raw[:, 8]

    def safe_div(a, b):
        # Handle arrays: return 0 where b is 0
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    # 9 Raw + 5 Engineered
    feats = [
        mq2, mq3, mq4, mq5, mq6, mq7, mq8, mq9, mq135,
        safe_div(mq3, mq4),          # MQ3_MQ4_Ratio
        safe_div(mq8, mq3),          # MQ8_MQ3_Ratio
        mq3 + mq135,                 # MQ3_MQ135_Sum
        safe_div(mq5, mq8),          # MQ5_MQ8_Ratio
        mq2+mq3+mq4+mq5+mq6+mq7+mq8+mq9+mq135,  # Total_Active_VOC
    ]
    return np.column_stack(feats)

# ── Load / Generate Data ──────────────────────────────────────────────────────
SMELL_CLASSES = [
    "banana", "blueberry", "grape", "green",
    "kiwi", "mushroom", "pear", "red",
    "strawberry", "tomato",
]

rng = np.random.default_rng(42)
X_parts, y_parts = [], []

for class_idx, smell in enumerate(SMELL_CLASSES):
    # Each class has a signature sensor from the HEURISTIC_MAP in app.py
    samples = rng.integers(100, 300, size=(150, 9))
    
    # Map class to a "dominant" sensor to make synthetic data realistic
    sensor_map = {
        "strawberry": 0, "banana": 1, "mushroom": 2, "grape": 3,
        "blueberry": 4, "tomato": 5, "kiwi": 6, "pear": 7, "green": 8
    }
    dom_idx = sensor_map.get(smell, class_idx % 9)
    
    # Elevate the dominant sensor reading
    samples[:, dom_idx] = rng.integers(600, 950, size=150)
    
    # Set MQ9 to 0 as per hardware constraint in app.py
    samples[:, 7] = 0.0 
    
    X_parts.append(samples)
    y_parts.extend([smell] * 150)

X_raw = np.vstack(X_parts).astype(float)
y = np.array(y_parts)

# Apply Feature Engineering
X = engineer_features_batch(X_raw)

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
print("\n-- Classification Report ---------------------------------------------")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Feature importance ────────────────────────────────────────────────────────
print("-- Feature Importances -----------------------------------------------")
for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
    bar = "#" * int(imp * 60)
    print(f"  {name:>15}  {imp:.4f}  {bar}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "smell_classifier.pkl")
label_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

with open(model_path, "wb") as f:
    pickle.dump(clf, f)
with open(label_path, "wb") as f:
    pickle.dump(le, f)

print(f"\n[OK] Model saved  ->  {model_path}")
print(f"[OK] Encoder saved ->  {label_path}")
print("\nRestart the Flask server (or POST to /api/reload_model) to use the model.")
