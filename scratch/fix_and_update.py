import pandas as pd
import numpy as np
import json

def fix_and_update():
    file_path = 'Good_apple.xlsx'
    # Read without header
    df = pd.read_excel(file_path, header=None)
    data = df.values.tolist()
    
    # If the first row contains 'alcohol', it might be from the previous failed run
    # where it was treated as a header.
    # Let's see what data[0] looks like.
    print(f"DEBUG: data[0] = {data[0]}")
    
    # Simple fix: if 'alcohol' is in the first row, we might need to clean up.
    # Actually, if I read with header=None, the first row is just the first row.
    
    # Let's just find the columns and make sure we have exactly what we want.
    # The original had 13 columns (0-12).
    # We want 15 columns (0-14).
    
    # Trim to 13 columns first if needed
    clean_data = [row[:13] for row in data]
    
    # Re-add alcohol
    clean_data[0].extend(["alcohol", None])
    clean_data[1].extend(["min", "max"])
    
    sensor_map = {
        "MQ2": (601, 1023),
        "MQ3": (601, 1023)
    }
    
    for i in range(2, len(clean_data)):
        sensor_name = clean_data[i][0]
        if sensor_name in sensor_map:
            clean_data[i].extend([601, 1023])
        else:
            clean_data[i].extend([None, None])
            
    # Save
    pd.DataFrame(clean_data).to_excel(file_path, index=False, header=False)
    print("Excel fixed and updated.")
    
    # Parse
    rules = {}
    smells = ["apple", "good tomato", "bad tomato", "good banana", "good potato", "bad banana", "alcohol"]
    for row_idx in range(2, len(clean_data)):
        sensor_name = clean_data[row_idx][0]
        if not isinstance(sensor_name, str): continue
        for smell_idx, smell in enumerate(smells):
            if smell not in rules: rules[smell] = {}
            min_val = clean_data[row_idx][1 + smell_idx * 2]
            max_val = clean_data[row_idx][2 + smell_idx * 2]
            try:
                if pd.isna(min_val): min_val = None
                else: min_val = float(min_val)
                if pd.isna(max_val): max_val = None
                else: max_val = float(max_val)
            except:
                min_val = max_val = None
            rules[smell][sensor_name] = {"min": min_val, "max": max_val}
            
    with open('excel_rules.json', 'w') as f:
        json.dump(rules, f, indent=4)
    print("Rules saved to excel_rules.json")

if __name__ == "__main__":
    fix_and_update()
