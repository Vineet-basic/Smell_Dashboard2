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
# Baseline readings captured during start or manual recalibration.
baseline_readings = {"MQ2": 0.0, "MQ3": 0.0, "MQ135": 0.0}

def apply_calibration(raw: dict) -> dict:
    """Returns delta values (raw - baseline)."""
    calibrated = {}
    for s in SENSOR_NAMES:
        calibrated[s] = max(0, raw.get(s, 0) - baseline_readings.get(s, 0))
    return calibrated

# ─── Sensor Configuration ─────────────────────────────────────────────────────
SENSOR_NAMES = ["MQ2", "MQ3", "MQ135"]

# ─── Labels & Classification ───────────────────────────────────────────────────
LABELS = ['Clean Air', 'Smoke', 'Alcohol', 'Activity Detected', 'Fresh', 'Ripe', 'Spoiling', 'Fermenting']
ACTIVE_MODE = "normalization" # "normalization" or "spoilage"

def get_normalized_readings(raw: dict) -> dict:
    """Returns values normalized between 0.0 and 1.0 based on baseline."""
    normalized = {}
    for s in SENSOR_NAMES:
        baseline = baseline_readings.get(s, 0)
        # Normalize: (Raw - Baseline) / (Max - Baseline)
        range_val = max(1, 1023 - baseline)
        val = max(0, raw.get(s, 0) - baseline)
        normalized[s] = round(val / range_val, 4)
    return normalized

def run_spoilage_prediction(raw_calibrated: dict) -> dict:
    """Specific logic for fruit spoilage based on gas signatures."""
    mq3 = raw_calibrated.get("MQ3", 0)
    mq135 = raw_calibrated.get("MQ135", 0)
    mq2 = raw_calibrated.get("MQ2", 0)
    
    # Normalization for the result dict
    normalized = get_normalized_readings(raw_calibrated)
    max_norm = max(normalized.values())
    dominant = max(normalized, key=normalized.get)
    
    if mq3 > 300 or mq135 > 250:
        label = "Fermenting"
        note = "High alcohol/gas levels indicating fermentation."
    elif mq3 > 150 or mq135 > 100:
        label = "Spoiling"
        note = "Significant gas increase; fruit may be rotting."
    elif mq3 > 40 or mq135 > 30:
        label = "Ripe"
        note = "Moderate activity; fruit is likely ripe."
    else:
        label = "Fresh"
        note = "Sensor values at baseline; fruit appears fresh."
        
    return {
        "label": label,
        "confidence": round(max_norm * 100, 2),
        "dominant_sensor": dominant,
        "note": note,
        "mode": "spoilage",
        "normalized": normalized
    }

def run_prediction(raw_calibrated: dict) -> dict:
    """
    Delegates to the active mode's predictor.
    """
    if ACTIVE_MODE == "spoilage":
        return run_spoilage_prediction(raw_calibrated)
    
    # Default Normalization Mode Logic
    normalized = get_normalized_readings(raw_calibrated)
    
    mq2_delta = raw_calibrated.get("MQ2", 0)
    mq3_delta = raw_calibrated.get("MQ3", 0)
    mq135_delta = raw_calibrated.get("MQ135", 0)
    
    # Find strongest normalized activity
    dominant_sensor = max(normalized, key=normalized.get)
    max_norm = normalized[dominant_sensor]
    
    # Specific User Rules (Deltas)
    if mq2_delta > 150:
        label = "Smoke"
        note  = f"Smoke detected (MQ2 Delta: {round(mq2_delta)})"
    elif mq3_delta > 30 or mq135_delta > 30:
        label = "Alcohol"
        note  = f"Alcohol detected (MQ3: {round(mq3_delta)}, MQ135: {round(mq135_delta)})"
    elif max_norm < 0.05: # Threshold for general detection
        label = "Clean Air"
        note  = "Sensors at baseline environment levels."
    else:
        label = "Activity Detected"
        note  = f"Significant increase on {dominant_sensor} ({round(max_norm*100, 1)}% normalized activity)."

    return {
        "label":           label,
        "confidence":      round(max_norm * 100, 2),
        "dominant_sensor": dominant_sensor,
        "note":            note,
        "mode":            "normalization",
        "normalized":      normalized
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
        
        self.is_calibrating = False
        self.calib_buffer   = []
        self.calib_target   = 10  # samples

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
            "connected":    self.running and self.ser is not None,
            "port":         self.port,
            "baud":         self.baud,
            "error":        self.error,
            "calibrating":  self.is_calibrating,
        }

    def start_calibration(self):
        with self.lock:
            self.is_calibrating = True
            self.calib_buffer = []
            print("[SERIAL] Calibration started...")

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
                            if self.is_calibrating:
                                with self.lock:
                                    self.calib_buffer.append(parsed)
                                    if len(self.calib_buffer) >= self.calib_target:
                                        self._finish_calibration()
                                continue # Skip prediction during calibration

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
            # MASTER: <MQ5>,<MQ135>,<MQ8>,<MQ3>  ||  SLAVE: <MQ7>,<MQ6>,<MQ2>,<MQ4>
            return {
                "MQ135": master_vals[1],
                "MQ3":   master_vals[3],
                "MQ2":   slave_vals[2],
            }
        except Exception:
            return None

    def _finish_calibration(self):
        global baseline_readings
        if not self.calib_buffer:
            self.is_calibrating = False
            return
            
        new_baseline = {s: 0.0 for s in SENSOR_NAMES}
        for entry in self.calib_buffer:
            for s in SENSOR_NAMES:
                new_baseline[s] += entry[s]
        
        for s in SENSOR_NAMES:
            new_baseline[s] /= len(self.calib_buffer)
            
        baseline_readings = new_baseline
        self.is_calibrating = False
        print(f"[SERIAL] Calibration complete. Baseline: {baseline_readings}")


serial_reader = SerialReader()

@app.route("/")
def index():
    return render_template("index.html", sensors=SENSOR_NAMES)


# ── Manual predict (from dashboard inputs) ─────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body: { "sensors": [MQ2, MQ3, MQ135] }
    """
    data = request.get_json(force=True)
    if "sensors" not in data:
        return jsonify({"error": "Missing 'sensors' key."}), 400

    vals = data["sensors"]
    if len(vals) != len(SENSOR_NAMES):
        return jsonify({"error": f"Expected {len(SENSOR_NAMES)} values, got {len(vals)}."}), 400
        
    try:
        vals = [float(v) for v in vals]
    except (ValueError, TypeError):
        return jsonify({"error": "All values must be numeric."}), 400

    raw = dict(zip(SENSOR_NAMES, vals))
    calibrated = apply_calibration(raw)
    result = run_prediction(calibrated)
    
    entry  = log_reading(calibrated, result)
    result["timestamp"] = entry["timestamp"]
    return jsonify(result)


# ── Live reading (from serial) ─────────────────────────────────────────────────
@app.route("/api/live", methods=["GET"])
def live():
    with serial_reader.lock:
        data = serial_reader.latest
    return jsonify({"data": data, "serial": serial_reader.status()})


# ─── Calibration & Mode ────────────────────────────────────────────────────────
@app.route("/api/mode", methods=["GET"])
def get_mode():
    return jsonify({
        "active_mode":    ACTIVE_MODE,
        "valid_modes":    ["normalization", "spoilage"],
        "descriptions": {
            "normalization": "General smoke and alcohol detection based on sensor deltas.",
            "spoilage": "Analyzes fruit state (Fresh, Ripe, Spoiling, Fermenting)."
        }
    })

@app.route("/api/mode", methods=["POST"])
def set_mode():
    global ACTIVE_MODE
    data = request.get_json(force=True)
    new_mode = data.get("mode")
    if new_mode in ["normalization", "spoilage"]:
        ACTIVE_MODE = new_mode
        return jsonify({"success": True, "active_mode": ACTIVE_MODE})
    return jsonify({"error": "Invalid mode"}), 400


@app.route("/api/calibrate", methods=["POST"])
def calibrate_sensors():
    if not serial_reader.running:
        return jsonify({"error": "Serial not connected."}), 400
    serial_reader.start_calibration()
    return jsonify({"success": True, "message": "Calibration started."})


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
    return jsonify({"sensors": SENSOR_NAMES})


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
        "active_mode":          ACTIVE_MODE,
        "labels":               LABELS,
        "sensors":              SENSOR_NAMES,
        "total_predictions":    len(recent_readings),
        "serial":               serial_reader.status(),
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Digital Smell Classifier - Flask Server")
    print(f"  Sensors  : {', '.join(SENSOR_NAMES)}")
    print(f"  Mode     : Normalization (Environment-Adaptive)")
    print("  URL      : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
