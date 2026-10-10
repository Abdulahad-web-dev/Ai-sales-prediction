import pandas as pd
import os
import requests

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    output_path = "data/Online_Retail.xlsx"
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset downloaded successfully and saved to {output_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_data()
