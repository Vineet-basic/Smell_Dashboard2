import pandas as pd
import numpy as np
import json

def parse_rules():
    file_path = 'Good_apple.xlsx'
    # Read without header to control column indices
    df = pd.read_excel(file_path, header=None)
    data = df.values.tolist()
    
    # Smells and their start column indices (indices 1, 3, 5, 7, 9, 11, 13, 15)
    # We'll use a fixed list since we know the structure now
    smells_info = [
        (1, "apple"),
        (3, "good tomato"),
        (5, "bad tomato"),
        (7, "good banana"),
        (9, "good potato"),
        (11, "bad banana"),
        (13, "alcohol"),
        (15, "smoke detected")
    ]
    
    rules = {}
    
    # Iterate through sensor rows (starting from row index 2)
    for row_idx in range(2, len(data)):
        sensor_name = data[row_idx][0]
        if not isinstance(sensor_name, str): continue
        
        for start_col, smell_name in smells_info:
            if smell_name not in rules:
                rules[smell_name] = {}
            
            # Check if index exists
            if start_col + 1 >= len(data[row_idx]):
                # If alcohol/smoke were added, they might be at the end
                # but let's be safe
                min_val = None
                max_val = None
            else:
                min_val = data[row_idx][start_col]
                max_val = data[row_idx][start_col + 1]
            
            try:
                if pd.isna(min_val): min_val = None
                else: min_val = float(min_val)
                
                if pd.isna(max_val): max_val = None
                else: max_val = float(max_val)
            except:
                min_val = max_val = None
                
            rules[smell_name][sensor_name] = {"min": min_val, "max": max_val}
            
    with open('excel_rules.json', 'w') as f:
        json.dump(rules, f, indent=4)
    print("Rules parsed and saved correctly.")

if __name__ == "__main__":
    parse_rules()
