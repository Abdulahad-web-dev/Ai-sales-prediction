import pandas as pd
import os

input_path = 'data/Online_Retail.xlsx'
output_path = 'data/Online_Retail.csv'

if os.path.exists(input_path):
    print("Converting Excel to CSV for faster processing...")
    df = pd.read_excel(input_path)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
else:
    print("Excel file not found.")
