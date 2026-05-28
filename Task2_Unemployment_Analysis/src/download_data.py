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
    print("           UNEMPLOYMENT ANALYSIS: DATA DOWNLOADER")
    print("=" * 60)
    
    # 1. Create data directory
    os.makedirs("data", exist_ok=True)
    
    # 2. Define files and URLs
    datasets = {
        "Unemployment in India.csv": [
            "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno2/main/Unemployment%20in%20India.csv",
            "https://raw.githubusercontent.com/deepanshudagdi/Unemployment-in-India/master/Unemployment%20in%20India.csv"
        ],
        "Unemployment_Rate_upto_11_2020.csv": [
            "https://raw.githubusercontent.com/jarif87/DataSets/main/Unemployment_Rate_upto_11_2020.csv",
            "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno2/main/Unemployment%20Rate%20upto%2011_2020.csv",
            "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno2/master/Unemployment%20Rate%20upto%2011_2020.csv",
            "https://raw.githubusercontent.com/deepanshudagdi/Unemployment-in-India/master/Unemployment%20Rate%20upto%2011_2020.csv",
            "https://raw.githubusercontent.com/deepanshudagdi/Unemployment-in-India/main/Unemployment%20Rate%20upto%2011_2020.csv"
        ]
    }
    
    # 3. Download files
    success_count = 0
    for filename, urls in datasets.items():
        destination = os.path.join("data", filename)
        print(f"\nProcessing '{filename}'...")
        
        # Try URLs in order until one succeeds
        for url in urls:
            if download_file(url, destination):
                success_count += 1
                break
            else:
                print(" -> Retrying with alternative URL...")
                
    print("\n" + "=" * 60)
    if success_count == 2:
        print(" SUCCESS: All datasets downloaded and prepared successfully!")
    else:
        print(f" WARNING: Only {success_count}/2 datasets were downloaded. Please check internet connection.")
    print("=" * 60)

if __name__ == "__main__":
    main()
