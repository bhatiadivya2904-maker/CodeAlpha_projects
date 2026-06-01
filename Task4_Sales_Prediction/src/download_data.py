import os
import requests

def download_file(url, destination):
    print(f"Downloading from: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f" -> Successfully saved to: {destination}")
        return True
    except Exception as e:
        print(f" -> Error downloading file: {e}")
        return False

def main():
    print("=" * 60)
    print("         SALES PREDICTION: DATASET DOWNLOADER")
    print("=" * 60)
    
    # 1. Create data directory
    os.makedirs("data", exist_ok=True)
    
    # 2. Define files and URLs
    url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
    destination = os.path.join("data", "advertising.csv")
    
    # 3. Download files
    success = download_file(url, destination)
                 
    print("\n" + "=" * 60)
    if success:
        print(" SUCCESS: Advertising dataset downloaded and prepared successfully!")
    else:
        print(" WARNING: Failed to download dataset. Please check your internet connection.")
    print("=" * 60)

if __name__ == "__main__":
    main()
