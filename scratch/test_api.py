import requests
import json

url = "http://127.0.0.1:5000/api/predict"
data = {"sensors": [324, 546, 470, 213, 506, 246, 253, 0, 208]}

try:
    response = requests.post(url, json=data)
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
