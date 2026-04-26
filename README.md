# Digital Smell Classifier – Project Structure

```
Smell_Dashboard/
├── app.py                  # Flask server  ← main entry point
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── model/                  # Auto-created after training
│   ├── smell_classifier.pkl
│   └── label_encoder.pkl
└── templates/
    └── index.html          # Interactive dashboard UI
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

> The server starts in **Demo mode** if no trained model is present.
> It uses a heuristic (dominant sensor) to predict the smell class.

### 3. (Optional) Train with your data
Edit `train_model.py` to load your CSV, then:
```bash
python train_model.py
```
After training, either restart the server or call:
```
POST /api/reload_model
```

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Dashboard UI |
| `POST` | `/api/predict` | Classify a sensor reading |
| `GET`  | `/api/sensors` | List sensor names |
| `GET`  | `/api/history` | Recent predictions |
| `GET`  | `/api/status`  | Health check / model status |
| `POST` | `/api/reload_model` | Hot-reload model without restart |

### POST `/api/predict` – Example

**Request**
```json
{
  "sensors": [350, 120, 80, 95, 70, 610, 55, 200, 310]
}
```
Sensor order: `MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135`

**Response**
```json
{
  "label": "Carbon Monoxide",
  "confidence": 94.5,
  "dominant_sensor": "MQ7",
  "note": "Prediction from trained model.",
  "mode": "model",
  "timestamp": "2026-04-25T17:55:00.123456"
}
```

---

## Sensor Risk Levels
| Colour | Sensors | Risk |
|--------|---------|------|
| 🔴 Red border on focus | MQ2, MQ7 | High |
| 🟡 Yellow | MQ3, MQ5, MQ6, MQ9 | Medium |
| 🟢 Green | MQ4, MQ8, MQ135 | Low |
