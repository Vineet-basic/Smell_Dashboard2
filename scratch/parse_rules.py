import pandas as pd
import json

def parse_excel_rules(file_path):
    df = pd.read_excel(file_path)
    
    # Extract smells from the first row (excluding the first column which is sensor name)
    # The first row in pandas might be the header or the first data row depending on how it's read.
    # Looking at the JSON output from previous step:
    # Row 0: {"Unnamed: 0":null,"Unnamed: 1":"apple","Unnamed: 2":null,"Unnamed: 3":"good tomato", ...}
    # Row 1: {"Unnamed: 0":null,"Unnamed: 1":"min","Unnamed: 2":"max", ...}
    
    # We can use the JSON we got earlier to be sure about the structure.
    # But let's try to be robust.
    
    data = df.values.tolist()
    headers = df.columns.tolist()
    
    # Smells are in the first row of data (index 0)
    smell_row = data[0]
    # Min/Max are in the second row of data (index 1)
    # Sensor names start from index 2
    
    smells = []
    for i in range(1, len(smell_row), 2):
        if i < len(smell_row) and isinstance(smell_row[i], str):
            smells.append(smell_row[i])
        elif i-1 >= 0 and isinstance(smell_row[i-1], str):
             # Handle cases where the smell name is only in the 'min' column
            smells.append(smell_row[i-1])
            
    # Better yet, let's use the actual columns
    rules = {}
    
    # Sensor rows start from data[2]
    for row_idx in range(2, len(data)):
        sensor_name = data[row_idx][0]
        if not isinstance(sensor_name, str):
            continue
            
        for smell_idx, smell in enumerate(smells):
            if smell not in rules:
                rules[smell] = {}
            
            min_val = data[row_idx][1 + smell_idx * 2]
            max_val = data[row_idx][2 + smell_idx * 2]
            
            # Convert to float if possible, otherwise skip or use None
            try:
                if pd.isna(min_val): min_val = None
                else: min_val = float(min_val)
                
                if pd.isna(max_val): max_val = None
                else: max_val = float(max_val)
            except (ValueError, TypeError):
                min_val = None
                max_val = None
                
            rules[smell][sensor_name] = {"min": min_val, "max": max_val}
            
    return rules

if __name__ == "__main__":
    rules = parse_excel_rules('Good_apple.xlsx')
    with open('excel_rules.json', 'w') as f:
        json.dump(rules, f, indent=4)
    print("Rules saved to excel_rules.json")
