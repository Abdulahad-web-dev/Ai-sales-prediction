import requests
import os

def download_file(url, filename):
    print(f"Starting download: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        
        downloaded = 0
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                if total_size > 0:
                    done = int(50 * downloaded / total_size)
                    print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB", end='')
                else:
                    print(f"\rDownloaded: {downloaded/1024/1024:.2f}MB", end='')
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    download_file(url, "data/Online_Retail.xlsx")
