import pandas as pd
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
output_path = "data/Online_Retail.xlsx"

if not os.path.exists("data"):
    os.makedirs("data")

print("Attempting to load data directly from URL using pandas...")
try:
    # This might take a while but pandas handles it well
    df = pd.read_excel(url)
    print("Data loaded successfully!")
    df.to_excel(output_path, index=False)
    print(f"Data saved to {output_path}")
except Exception as e:
    print(f"Error: {e}")
