import pandas as pd
import json
import sys

try:
    df = pd.read_excel('Good_apple.xlsx')
    print(df.to_json(orient='records'))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
