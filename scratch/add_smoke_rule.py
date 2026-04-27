import pandas as pd
import numpy as np
import json

def add_smoke_detected_rule():
    file_path = 'Good_apple.xlsx'
    df = pd.read_excel(file_path, header=None)
    data = df.values.tolist()
    
    # Check if 'smoke detected' already exists
    if "smoke detected" not in data[0]:
        # Add to first row
        data[0].extend(["smoke detected", None])
        # Add to second row
        data[1].extend(["min", "max"])
        
        sensor_map = {
            "MQ2": (0, 399),
            "MQ3": (581, 1023)
        }
        
        for i in range(2, len(data)):
            sensor_name = data[i][0]
            if sensor_name in sensor_map:
                data[i].extend([sensor_map[sensor_name][0], sensor_map[sensor_name][1]])
            else:
                data[i].extend([None, None])
                
        # Save back to excel
        pd.DataFrame(data).to_excel(file_path, index=False, header=False)
        print("Excel updated with smoke detected rule.")
    else:
        print("Smoke detected rule already exists.")

    # Re-parse everything from the updated excel to excel_rules.json
    # We need to know all the smells currently in the excel
    smell_row = data[0]
    smells = []
    for i in range(1, len(smell_row), 2):
        s = smell_row[i]
        if pd.isna(s): s = smell_row[i-1]
        if not pd.isna(s): smells.append(str(s))
    
    rules = {}
    for row_idx in range(2, len(data)):
        sensor_name = data[row_idx][0]
        if not isinstance(sensor_name, str): continue
        for smell_idx, smell in enumerate(smells):
            if smell not in rules: rules[smell] = {}
            min_val = data[row_idx][1 + smell_idx * 2]
            max_val = data[row_idx][2 + smell_idx * 2]
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
    add_smoke_detected_rule()
