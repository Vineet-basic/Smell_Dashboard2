import pandas as pd
import numpy as np

def update_excel_with_alcohol(file_path):
    df = pd.read_excel(file_path, header=None)
    data = df.values.tolist()
    
    # Add to first row
    data[0].extend(["alcohol", None])
    # Add to second row
    data[1].extend(["min", "max"])
    
    sensor_map = {
        "MQ2": (601, 1023),
        "MQ3": (601, 1023)
    }
    
    for i in range(2, len(data)):
        sensor_name = data[i][0]
        if sensor_name in sensor_map:
            data[i].extend([sensor_map[sensor_name][0], sensor_map[sensor_name][1]])
        else:
            data[i].extend([None, None])
            
    # Save back to excel
    new_df = pd.DataFrame(data)
    new_df.to_excel(file_path, index=False, header=False)
    print(f"Updated {file_path} with alcohol rule.")

if __name__ == "__main__":
    update_excel_with_alcohol('Good_apple.xlsx')
