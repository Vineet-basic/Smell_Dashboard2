"""
app.py  –  Digital Smell Classifier · Flask Server
====================================================
Serial format expected from Master Arduino:
    MASTER: <MQ2>,<MQ3>,<MQ4>,<MQ5>  ||  SLAVE: <MQ6>,<MQ7>,<MQ8>,<MQ135>

MQ9 is set to 0 (not wired in current hardware).

Feature engineering mirrors your prediction notebook exactly:
    MQ3_MQ4_Ratio, MQ8_MQ3_Ratio, MQ3_MQ135_Sum,
    MQ5_MQ8_Ratio, Total_Active_VOC
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
import threading
import time
from datetime import datetime

import serial
import serial.tools.list_ports

app = Flask(__name__)

# ─── Sensor Configuration ─────────────────────────────────────────────────────
SENSOR_NAMES = ["MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9", "MQ135"]

# Feature order MUST match what the model was trained on
FEATURE_NAMES = [
    "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9", "MQ135",
    "MQ3_MQ4_Ratio", "MQ8_MQ3_Ratio", "MQ3_MQ135_Sum",
    "MQ5_MQ8_Ratio", "Total_Active_VOC",
]

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

# ─── Prediction helper ─────────────────────────────────────────────────────────
# Real class labels from trained model
LABELS = [
    'banana', 'blueberry', 'grape', 'green',
    'kiwi', 'mushroom', 'pear', 'red',
    'strawberry', 'tomato',
]

# Dominant-sensor → (label, emoji note) used only in DEMO mode (no model loaded)
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

def run_prediction(raw: dict) -> dict:
    """
    Accepts a dict of raw sensor values.
    Returns prediction result dict.
    """
    dominant_sensor = SENSOR_NAMES[int(np.argmax([raw[s] for s in SENSOR_NAMES]))]

    if model is not None:
        feats = engineer_features(raw).reshape(1, -1)
        label = str(model.predict(feats)[0])
        proba = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
        confidence = round(float(np.max(proba)) * 100, 2) if proba is not None else None
        note = "Prediction from trained model."
        mode = "model"
    else:
        smell, note = HEURISTIC_MAP.get(dominant_sensor, ("Unknown", "No clear signature"))
        label = smell
        confidence = round(min(raw[dominant_sensor] / 1023.0, 1.0) * 100, 2)
        mode = "demo"

    return {
        "label": label,
        "confidence": confidence,
        "dominant_sensor": dominant_sensor,
        "note": note,
        "mode": mode,
    }

# ─── In-memory history ─────────────────────────────────────────────────────────
recent_readings = []
MAX_READINGS = 100

def log_reading(raw: dict, result: dict, source: str = "manual"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "sensor_values": raw,
        "source": source,
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
        MASTER: <MQ2>,<MQ3>,<MQ4>,<MQ5>  ||  SLAVE: <MQ6>,<MQ7>,<MQ8>,<MQ135>
    """
    def __init__(self):
        self.ser        = None
        self.running    = False
        self.thread     = None
        self.port       = None
        self.baud       = 9600
        self.latest     = None          # latest parsed + predicted reading
        self.error      = None          # last error string
        self.raw_lines  = []            # last 50 raw serial lines (for debug)
        self.lock       = threading.Lock()

    # ── Public interface ───────────────────────────────────────────────────────
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

    # ── Background loop ────────────────────────────────────────────────────────
    def _loop(self):
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
                            result  = run_prediction(parsed)
                            entry   = log_reading(parsed, result, source="serial")
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
        """
        Parses:
            MASTER: 350,120,80,95  ||  SLAVE: 610,55,200,310
        Maps values to:
            MQ2,MQ3,MQ4,MQ5 (master)  +  MQ6,MQ7,MQ8,MQ135 (slave)
        MQ9 defaults to 0.
        """
        try:
            if "MASTER:" not in line or "SLAVE:" not in line:
                return None
            master_part, slave_part = line.split("||")
            master_vals = [float(x) for x in master_part.replace("MASTER:", "").strip().split(",")]
            slave_vals  = [float(x) for x in slave_part.replace("SLAVE:", "").strip().split(",")]

            if len(master_vals) < 4 or len(slave_vals) < 4:
                return None

            return {
                "MQ2":   master_vals[0],
                "MQ3":   master_vals[1],
                "MQ4":   master_vals[2],
                "MQ5":   master_vals[3],
                "MQ6":   slave_vals[0],
                "MQ7":   slave_vals[1],
                "MQ8":   slave_vals[2],
                "MQ9":   0.0,            # not wired; add when available
                "MQ135": slave_vals[3],
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

    raw    = dict(zip(SENSOR_NAMES, vals))
    result = run_prediction(raw)
    entry  = log_reading(raw, result)
    result["timestamp"] = entry["timestamp"]
    return jsonify(result)


# ── Live reading (from serial) ─────────────────────────────────────────────────
@app.route("/api/live", methods=["GET"])
def live():
    """Returns the latest serial reading + prediction (or null if none yet)."""
    with serial_reader.lock:
        data = serial_reader.latest
    return jsonify({"data": data, "serial": serial_reader.status()})


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
    """Last 50 raw lines received from Arduino (useful for debugging)."""
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


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status":            "ok",
        "model_loaded":      model is not None,
        "mode":              "model" if model is not None else "demo",
        "labels":            LABELS,
        "sensors":           SENSOR_NAMES,
        "total_predictions": len(recent_readings),
        "serial":            serial_reader.status(),
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
    print("  Digital Smell Classifier – Flask Server")
    print(f"  Sensors  : {', '.join(SENSOR_NAMES)}")
    print(f"  Features : {len(FEATURE_NAMES)} (with engineered)")
    print(f"  Model    : {'Loaded ✓' if model else 'Demo mode (no model.pkl found)'}")
    print("  URL      : http://127.0.0.1:5000")
    print("=" * 60)
    # use_reloader=False prevents the serial thread from being killed on reload
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
