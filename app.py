"""
app.py  –  Digital Smell Classifier · Flask Server
====================================================
Serial format expected from Master Arduino:
    MASTER: <MQ5>,<MQ135>,<MQ8>,<MQ3>  ||  SLAVE: <MQ7>,<MQ6>,<MQ2>,<MQ4>

MQ9 is set to 0 (not wired in current hardware).

Feature engineering remains consistent with the model training:
    MQ3_MQ4_Ratio, MQ8_MQ3_Ratio, MQ3_MQ135_Sum,
    MQ5_MQ8_Ratio, Total_Active_VOC

Classification Modes
--------------------
  model     – Pre-trained ML model (.pkl) with engineered features
  heuristic – Dominant-sensor rule-based mapping (works without model)
  threshold – Configurable sensor-value threshold rules
"""

from flask import Flask, request, jsonify, render_template, Response
import numpy as np
import joblib
import os
import threading
import time
import io
import csv
from datetime import datetime
import json

import serial
import serial.tools.list_ports

app = Flask(__name__)

# ─── Calibration Settings ─────────────────────────────────────────────────────
# No calibration currently applied. Readings are displayed as raw ADC values (0-1023).

def apply_calibration(raw: dict) -> dict:
    """Returns raw values without modification."""
    return raw

# ─── Sensor Configuration ─────────────────────────────────────────────────────
SENSOR_NAMES = ["MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9", "MQ135"]

# Feature order MUST match what the model was trained on
FEATURE_NAMES = [
    "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9", "MQ135",
    "MQ3_MQ4_Ratio", "MQ8_MQ3_Ratio", "MQ3_MQ135_Sum",
    "MQ5_MQ8_Ratio", "Total_Active_VOC",
]

# ─── Classification Mode State ─────────────────────────────────────────────────
VALID_MODES = ["model", "heuristic", "threshold", "spoilage", "excel"]
active_classification_mode = "excel"   # Default to excel mode as requested

def get_effective_mode() -> str:
    """Returns the active mode, downgrading 'model' → 'heuristic' if no model loaded."""
    if active_classification_mode == "model" and model is None:
        return "heuristic"
    return active_classification_mode

# ─── Model paths (checks both model/ subdir and project root) ──────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(_BASE, "model", "model.pkl"),
    os.path.join(_BASE, "model.pkl"),
    os.path.join(_BASE, "model", "smell_classifier.pkl"),
]

model = None

def load_model():
    global model
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"[INFO] Model loaded from: {path}")
            return
    print("[WARN] No model file found – running in DEMO mode.")

load_model()

# ─── Excel Rules Loader ────────────────────────────────────────────────────────
excel_rules = {}
def load_excel_rules():
    global excel_rules
    path = os.path.join(_BASE, "excel_rules.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                excel_rules = json.load(f)
            print(f"[INFO] Excel rules loaded: {list(excel_rules.keys())}")
        except Exception as e:
            print(f"[ERROR] Failed to load excel_rules.json: {e}")
    else:
        print("[WARN] excel_rules.json not found.")

load_excel_rules()

# ─── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(raw: dict) -> np.ndarray:
    """
    raw: dict with keys MQ2..MQ135 (float values)
    Returns a 1-D numpy array of 14 features in FEATURE_NAMES order.
    Handles division-by-zero gracefully.
    """
    mq2   = raw["MQ2"]
    mq3   = raw["MQ3"]
    mq4   = raw["MQ4"]
    mq5   = raw["MQ5"]
    mq6   = raw["MQ6"]
    mq7   = raw["MQ7"]
    mq8   = raw["MQ8"]
    mq9   = raw["MQ9"]
    mq135 = raw["MQ135"]

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    feats = [
        mq2, mq3, mq4, mq5, mq6, mq7, mq8, mq9, mq135,
        safe_div(mq3, mq4),          # MQ3_MQ4_Ratio
        safe_div(mq8, mq3),          # MQ8_MQ3_Ratio
        mq3 + mq135,                 # MQ3_MQ135_Sum
        safe_div(mq5, mq8),          # MQ5_MQ8_Ratio
        mq2+mq3+mq4+mq5+mq6+mq7+mq8+mq9+mq135,  # Total_Active_VOC
    ]
    return np.array(feats, dtype=float)

# ─── Labels & Heuristics ───────────────────────────────────────────────────────
LABELS = [
    'banana', 'blueberry', 'grape', 'green',
    'kiwi', 'mushroom', 'pear', 'red',
    'strawberry', 'tomato',
    'apple', 'good tomato', 'bad tomato', 
    'good banana', 'bad banana', 'good potato',
    'alcohol', 'smoke detected'
]

# Dominant-sensor → label (used in heuristic mode)
HEURISTIC_MAP = {
    "MQ2":   ("strawberry",  "🍓 Sweet fruity VOC signature"),
    "MQ3":   ("banana",      "🍌 Ester-rich banana aroma"),
    "MQ4":   ("mushroom",    "🍄 Earthy / fungal scent"),
    "MQ5":   ("grape",       "🍇 Fermentation-like VOC"),
    "MQ6":   ("blueberry",   "🫐 Light berry aroma"),
    "MQ7":   ("tomato",      "🍅 Acidic / aldehyde profile"),
    "MQ8":   ("kiwi",        "🥝 Tart aromatic compound"),
    "MQ9":   ("pear",        "🍐 Mild ester signature"),
    "MQ135": ("green",       "🥬 Chlorophyll / green note"),
}

# ─── Threshold Rules ──────────────────────────────────────────────────────────
# Each rule: { sensor, min, max, label, note }
# Rules are evaluated in order; first match wins.
# Default rules give a sensible starting point – can be updated via /api/threshold/rules
threshold_rules = [
    {"sensor": "MQ3",  "min": 600, "max": 1023, "label": "banana",     "note": "High MQ3 (ester-rich vapor)"},
    {"sensor": "MQ2",  "min": 700, "max": 1023, "label": "strawberry",  "note": "High MQ2 (sweet VOC)"},
    {"sensor": "MQ5",  "min": 500, "max": 1023, "label": "grape",       "note": "High MQ5 (fermented)"},
    {"sensor": "MQ6",  "min": 450, "max": 1023, "label": "blueberry",   "note": "High MQ6 (light berry)"},
    {"sensor": "MQ7",  "min": 500, "max": 1023, "label": "tomato",      "note": "High MQ7 (aldehyde profile)"},
    {"sensor": "MQ8",  "min": 400, "max": 1023, "label": "kiwi",        "note": "High MQ8 (tart compound)"},
    {"sensor": "MQ4",  "min": 400, "max": 1023, "label": "mushroom",    "note": "High MQ4 (earthy)"},
    {"sensor": "MQ135","min": 400, "max": 1023, "label": "green",       "note": "High MQ135 (green note)"},
    {"sensor": "MQ9",  "min": 300, "max": 1023, "label": "pear",        "note": "High MQ9 (mild ester)"},
]

def run_threshold(raw: dict, dominant_sensor: str) -> dict:
    """Evaluate threshold rules; return first match or fallback to dominant sensor."""
    for rule in threshold_rules:
        v = raw.get(rule["sensor"], 0)
        if rule["min"] <= v <= rule["max"]:
            confidence = round(min((v - rule["min"]) / max(rule["max"] - rule["min"], 1), 1.0) * 100, 2)
            return {
                "label":           rule["label"],
                "confidence":      confidence,
                "dominant_sensor": rule["sensor"],
                "note":            rule["note"] + f"  [Threshold rule: {rule['sensor']} ∈ [{rule['min']}, {rule['max']}]]",
                "mode":            "threshold",
                "rule_matched":    rule,
            }
    # No rule fired – fall back to dominant sensor
    smell, note = HEURISTIC_MAP.get(dominant_sensor, ("Unknown", "No matching rule"))
    confidence = round(min(raw[dominant_sensor] / 1023.0, 1.0) * 100, 2)
    return {
        "label":           smell,
        "confidence":      confidence,
        "dominant_sensor": dominant_sensor,
        "note":            note + "  [No threshold rule matched – dominant sensor fallback]",
        "mode":            "threshold",
        "rule_matched":    None,
    }

# ─── Spoilage Detection Logic ──────────────────────────────────────────────────
def run_spoilage(raw: dict) -> dict:
    """
    Analyzes MQ135 (Ammonia/VOC), MQ3 (Alcohol), and MQ4 (Methane)
    to detect food decomposition or fruit fermentation.
    Returns: label, confidence, dominant_sensor, note, mode, stage
    """
    mq3   = raw.get("MQ3", 0)
    mq135 = raw.get("MQ135", 0)
    mq4   = raw.get("MQ4", 0)
    total = sum(raw.values())
    
    dominant_sensor = max(raw, key=raw.get)
    stage = "Stable"
    
    # 1. Critical Spoilage (High Ammonia or Methane)
    if mq135 > 600 or mq4 > 500:
        label = "Spoiled / Toxic"
        stage = "Danger"
        note = "🚨 CRITICAL: High decomposition gases. Potential health risk."
        conf = min((max(mq135, mq4) / 1023.0) * 100 + 10, 100)
    
    # 2. Early Decomposition
    elif mq135 > 400 or mq4 > 350:
        label = "Spoiling"
        stage = "Warning"
        note = "⚠️ WARNING: Rising ammonia/methane levels detected (organic decay)."
        conf = 75.0 + (max(mq135, mq4) - 350) / 2
    
    # 3. Fermentation (High Alcohol)
    elif mq3 > 550:
        label = "Fermenting"
        stage = "Warning"
        note = "🍎 OVER-RIPE: Significant alcohol/ester levels detected."
        conf = min((mq3 / 1023.0) * 100 + 5, 100)
    
    # 4. Ripening
    elif 300 < mq3 <= 550:
        label = "Ripe / Aromatic"
        stage = "Fresh"
        note = "🍏 RIPE: High aromatic signature, perfect for consumption."
        conf = 90.0
        
    # 5. Fresh / Normal
    elif 100 < mq135 <= 300:
        label = "Fresh"
        stage = "Fresh"
        note = "🥬 FRESH: Standard VOC levels for fresh organic matter."
        conf = 85.0
        
    # 6. Low levels
    else:
        label = "Clean / Baseline"
        stage = "Stable"
        note = "✅ CLEAN: No significant organic gases detected."
        conf = max(0, 100.0 - (total / 5000.0 * 100))
        
    return {
        "label":           label,
        "confidence":      round(conf, 2),
        "dominant_sensor": dominant_sensor,
        "note":            note,
        "mode":            "spoilage",
        "stage":           stage,
    }

# ─── Excel-based Prediction ────────────────────────────────────────────────────
def run_excel_prediction(raw: dict) -> dict:
    """
    Compare raw sensor values against min/max ranges defined in excel_rules.json.
    Returns the best matching smell based on the number of sensors within range.
    """
    best_smell = "Unknown"
    best_score = 0
    best_note = "No matching range found in Excel data."
    
    if not excel_rules:
        return {
            "label": "Error",
            "confidence": 0,
            "dominant_sensor": max(raw, key=raw.get),
            "note": "Excel rules not loaded.",
            "mode": "excel"
        }

    for smell, sensors in excel_rules.items():
        matches = 0
        total_sensors = 0
        match_details = []
        
        for s_name, range_vals in sensors.items():
            if s_name not in raw: continue
            if range_vals["min"] is None or range_vals["max"] is None:
                continue
            
            total_sensors += 1
            val = raw[s_name]
            if range_vals["min"] <= val <= range_vals["max"]:
                matches += 1
                match_details.append(f"{s_name} OK")
            else:
                # Add a bit of info about how far off it is?
                pass
        
        if total_sensors > 0:
            # Score is percentage of sensors in range
            score = (matches / total_sensors) * 100
            
            # Simple tie-breaker: if scores are equal, we could use distance to range centers
            if score > best_score:
                best_score = score
                best_smell = smell
                best_note = f"Matched {matches}/{total_sensors} sensors from Excel rules. ({', '.join(match_details)})"
                # Potential tie-breaker logic could go here
                pass
        

    return {
        "label":           best_smell,
        "confidence":      round(best_score, 2),
        "dominant_sensor": max(raw, key=raw.get),
        "note":            best_note,
        "mode":            "excel",
    }

# ─── Prediction dispatcher ─────────────────────────────────────────────────────
def run_prediction(raw: dict) -> dict:
    """
    Dispatches to the currently active classification mode.
    Returns a result dict always containing:
        label, confidence, dominant_sensor, note, mode
    """
    dominant_sensor = SENSOR_NAMES[int(np.argmax([raw[s] for s in SENSOR_NAMES]))]
    mode = get_effective_mode()

    if mode == "model":
        feats  = engineer_features(raw).reshape(1, -1)
        label  = str(model.predict(feats)[0])
        proba  = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
        conf   = round(float(np.max(proba)) * 100, 2) if proba is not None else None
        note   = "ML model prediction (trained classifier)."
        return {
            "label":           label,
            "confidence":      conf,
            "dominant_sensor": dominant_sensor,
            "note":            note,
            "mode":            "model",
        }

    elif mode == "threshold":
        return run_threshold(raw, dominant_sensor)

    elif mode == "spoilage":
        return run_spoilage(raw)

    elif mode == "excel":
        return run_excel_prediction(raw)

    else:  # heuristic
        smell, note = HEURISTIC_MAP.get(dominant_sensor, ("Unknown", "No clear signature"))
        conf = round(min(raw[dominant_sensor] / 1023.0, 1.0) * 100, 2)
        return {
            "label":           smell,
            "confidence":      conf,
            "dominant_sensor": dominant_sensor,
            "note":            note,
            "mode":            "heuristic",
        }

# ─── In-memory history ─────────────────────────────────────────────────────────
recent_readings = []
MAX_READINGS = 100

def log_reading(raw: dict, result: dict, source: str = "manual"):
    entry = {
        "timestamp":     datetime.now().isoformat(),
        "sensor_values": raw,
        "source":        source,
        **result,
    }
    recent_readings.append(entry)
    if len(recent_readings) > MAX_READINGS:
        recent_readings.pop(0)
    return entry

# ─── Serial Reader ─────────────────────────────────────────────────────────────
class SerialReader:
    """
    Background thread that reads from the master Arduino and parses:
        MASTER: <MQ5>,<MQ135>,<MQ8>,<MQ3>  ||  SLAVE: <MQ7>,<MQ6>,<MQ2>,<MQ4>
    """
    def __init__(self):
        self.ser        = None
        self.running    = False
        self.thread     = None
        self.port       = None
        self.baud       = 9600
        self.latest     = None
        self.error      = None
        self.raw_lines  = []
        self.lock       = threading.Lock()

    def connect(self, port: str, baud: int = 9600) -> bool:
        if self.running:
            self.disconnect()
        try:
            self.ser     = serial.Serial(port, baud, timeout=2)
            self.port    = port
            self.baud    = baud
            self.running = True
            self.error   = None
            self.thread  = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            print(f"[SERIAL] Connected → {port} @ {baud}")
            return True
        except serial.SerialException as e:
            self.error = str(e)
            print(f"[SERIAL] Connect failed: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser  = None
        self.port = None
        print("[SERIAL] Disconnected.")

    def status(self) -> dict:
        return {
            "connected": self.running and self.ser is not None,
            "port":      self.port,
            "baud":      self.baud,
            "error":     self.error,
        }

    def _loop(self):
        prediction_buffer = [] # List of (timestamp, result_dict)
        while self.running:
            try:
                if self.ser and self.ser.is_open:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        with self.lock:
                            self.raw_lines.append(line)
                            if len(self.raw_lines) > 50:
                                self.raw_lines.pop(0)
                        
                        parsed = self._parse(line)
                        if parsed:
                            # Apply calibration
                            calibrated = apply_calibration(parsed)
                            # Get raw prediction
                            raw_result = run_prediction(calibrated)
                            
                            now = time.time()
                            prediction_buffer.append((now, raw_result))
                            # Keep only last 15 seconds
                            prediction_buffer = [p for p in prediction_buffer if now - p[0] <= 15]
                            
                            # Perform voting
                            counts = {}
                            for _, r in prediction_buffer:
                                lbl = r['label']
                                counts[lbl] = counts.get(lbl, 0) + 1
                            
                            # Find winner
                            winner_label = max(counts, key=counts.get)
                            
                            # Construct stable result
                            stable_result = raw_result.copy()
                            stable_result['label'] = winner_label
                            stable_result['raw_label'] = raw_result['label'] # Current instant prediction
                            stable_result['is_stable'] = len(prediction_buffer) > 5 # Need at least some samples
                            stable_result['samples'] = len(prediction_buffer)
                            stable_result['note'] = f"Stabilized result (15s window). Most frequent: {winner_label} ({counts[winner_label]}/{len(prediction_buffer)} samples)"
                            
                            # Log to console
                            ts = datetime.now().strftime("%H:%M:%S")
                            print(f"[{ts}] RAW: {raw_result['label']} | STABLE: {winner_label} ({counts[winner_label]}/{len(prediction_buffer)})")
                            import sys
                            sys.stdout.flush()
                            
                            entry = log_reading(calibrated, stable_result, source="serial")
                            with self.lock:
                                self.latest = entry
            except serial.SerialException as e:
                self.error   = str(e)
                self.running = False
                print(f"[SERIAL] Lost connection: {e}")
                break
            except Exception as e:
                print(f"[SERIAL] Parse error: {e}")
                time.sleep(0.1)

    def _parse(self, line: str) -> dict | None:
        try:
            if "MASTER:" not in line or "SLAVE:" not in line:
                return None
            master_part, slave_part = line.split("||")
            master_vals = [float(x) for x in master_part.replace("MASTER:", "").strip().split(",")]
            slave_vals  = [float(x) for x in slave_part.replace("SLAVE:", "").strip().split(",")]
            if len(master_vals) < 4 or len(slave_vals) < 4:
                return None
            return {
                "MQ5":   master_vals[0],
                "MQ135": master_vals[1],
                "MQ8":   master_vals[2],
                "MQ3":   master_vals[3],
                "MQ7":   slave_vals[0],
                "MQ6":   slave_vals[1],
                "MQ2":   slave_vals[2],
                "MQ4":   slave_vals[3],
                "MQ9":   0.0,
            }
        except Exception:
            return None


serial_reader = SerialReader()

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", sensors=SENSOR_NAMES)


# ── Manual predict (from dashboard inputs) ─────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body: { "sensors": [MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135] }
    """
    data = request.get_json(force=True)
    if "sensors" not in data:
        return jsonify({"error": "Missing 'sensors' key."}), 400

    vals = data["sensors"]
    if len(vals) != len(SENSOR_NAMES):
        return jsonify({"error": f"Expected {len(SENSOR_NAMES)} values, got {len(vals)}.",
                        "expected_order": SENSOR_NAMES}), 400
    try:
        vals = [float(v) for v in vals]
    except (ValueError, TypeError):
        return jsonify({"error": "All values must be numeric."}), 400

    raw = dict(zip(SENSOR_NAMES, vals))
    calibrated = apply_calibration(raw)
    result = run_prediction(calibrated)
    
    # Log to console
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] MANUAL PREDICT: {result['label']} ({result['confidence']}%) | Dominant: {result['dominant_sensor']}")
    print(f"[{ts}] SENSORS: {calibrated}")
    import sys
    sys.stdout.flush()
    
    entry  = log_reading(calibrated, result)
    result["timestamp"] = entry["timestamp"]
    return jsonify(result)


# ── Live reading (from serial) ─────────────────────────────────────────────────
@app.route("/api/live", methods=["GET"])
def live():
    with serial_reader.lock:
        data = serial_reader.latest
    return jsonify({"data": data, "serial": serial_reader.status()})


# ── Classification mode management ────────────────────────────────────────────
@app.route("/api/mode", methods=["GET"])
def get_mode():
    return jsonify({
        "active_mode":    active_classification_mode,
        "effective_mode": get_effective_mode(),
        "valid_modes":    VALID_MODES,
        "model_loaded":   model is not None,
        "descriptions": {
            "model":     "Pre-trained ML classifier with engineered features",
            "heuristic": "Dominant-sensor rule-based mapping (no model needed)",
            "threshold": "Configurable value-range rules (editable via /api/threshold/rules)",
            "spoilage":  "Specialized detection for food rot, fermentation, and freshness",
            "excel":     "Predictions based on recorded min/max values from Excel file",
        }
    })


@app.route("/api/mode", methods=["POST"])
def set_mode():
    global active_classification_mode
    data = request.get_json(force=True)
    new_mode = data.get("mode", "").strip().lower()
    if new_mode not in VALID_MODES:
        return jsonify({"error": f"Invalid mode. Choose from: {VALID_MODES}"}), 400
    active_classification_mode = new_mode
    print(f"[INFO] Classification mode changed → {new_mode}")
    return jsonify({
        "success":        True,
        "active_mode":    active_classification_mode,
        "effective_mode": get_effective_mode(),
    })


# ── Threshold rule management ──────────────────────────────────────────────────
@app.route("/api/threshold/rules", methods=["GET"])
def get_threshold_rules():
    return jsonify({"rules": threshold_rules})


@app.route("/api/threshold/rules", methods=["POST"])
def set_threshold_rules():
    """
    Replace all threshold rules.
    Body: { "rules": [ { "sensor": "MQ3", "min": 600, "max": 1023,
                          "label": "banana", "note": "..." }, ... ] }
    """
    global threshold_rules
    data  = request.get_json(force=True)
    rules = data.get("rules", [])
    errors = []
    for i, r in enumerate(rules):
        for k in ("sensor", "min", "max", "label"):
            if k not in r:
                errors.append(f"Rule {i}: missing field '{k}'")
        if r.get("sensor") not in SENSOR_NAMES:
            errors.append(f"Rule {i}: unknown sensor '{r.get('sensor')}'")
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    threshold_rules = rules
    return jsonify({"success": True, "rules": threshold_rules})


@app.route("/api/threshold/rules/<int:index>", methods=["DELETE"])
def delete_threshold_rule(index: int):
    global threshold_rules
    if index < 0 or index >= len(threshold_rules):
        return jsonify({"error": "Index out of range"}), 404
    removed = threshold_rules.pop(index)
    return jsonify({"success": True, "removed": removed, "rules": threshold_rules})


# ── Serial management ──────────────────────────────────────────────────────────
@app.route("/api/serial/ports", methods=["GET"])
def list_ports():
    ports = [
        {"port": p.device, "description": p.description}
        for p in serial.tools.list_ports.comports()
    ]
    return jsonify({"ports": ports})


@app.route("/api/serial/connect", methods=["POST"])
def serial_connect():
    data = request.get_json(force=True)
    port = data.get("port", "")
    baud = int(data.get("baud", 9600))
    if not port:
        return jsonify({"error": "Port is required."}), 400
    ok = serial_reader.connect(port, baud)
    return jsonify({"success": ok, "error": serial_reader.error, "serial": serial_reader.status()})


@app.route("/api/serial/disconnect", methods=["POST"])
def serial_disconnect():
    serial_reader.disconnect()
    return jsonify({"success": True, "serial": serial_reader.status()})


@app.route("/api/serial/raw", methods=["GET"])
def serial_raw():
    with serial_reader.lock:
        lines = list(serial_reader.raw_lines)
    return jsonify({"lines": lines[-20:]})


# ── Common endpoints ───────────────────────────────────────────────────────────
@app.route("/api/sensors", methods=["GET"])
def get_sensors():
    return jsonify({"sensors": SENSOR_NAMES, "features": FEATURE_NAMES})


@app.route("/api/history", methods=["GET"])
def get_history():
    limit = min(int(request.args.get("limit", 20)), MAX_READINGS)
    return jsonify({"history": recent_readings[-limit:]})


@app.route("/api/history/export", methods=["GET"])
def export_history():
    """Returns the history as a CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    headers = ["timestamp", "label", "confidence", "dominant_sensor", "mode", "source"] + SENSOR_NAMES
    writer.writerow(headers)
    
    for r in recent_readings:
        row = [
            r["timestamp"],
            r["label"],
            r["confidence"],
            r["dominant_sensor"],
            r["mode"],
            r.get("source", "manual")
        ]
        # Append all sensor values
        for s in SENSOR_NAMES:
            row.append(r["sensor_values"].get(s, 0))
        writer.writerow(row)
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=smell_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status":               "ok",
        "model_loaded":         model is not None,
        "active_mode":          active_classification_mode,
        "effective_mode":       get_effective_mode(),
        "labels":               LABELS,
        "sensors":              SENSOR_NAMES,
        "total_predictions":    len(recent_readings),
        "serial":               serial_reader.status(),
        "threshold_rule_count": len(threshold_rules),
    })


@app.route("/api/reload_model", methods=["POST"])
def reload_model():
    load_model()
    return jsonify({
        "success":      True,
        "model_loaded": model is not None,
        "message":      "Model reloaded." if model else "No model file found.",
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Digital Smell Classifier - Flask Server")
    print(f"  Sensors  : {', '.join(SENSOR_NAMES)}")
    print(f"  Features : {len(FEATURE_NAMES)} (with engineered)")
    print(f"  Model    : {'Loaded OK' if model else 'Demo mode (no model.pkl found)'}")
    print(f"  Mode     : {active_classification_mode}")
    print("  URL      : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
