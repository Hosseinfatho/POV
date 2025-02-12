import requests
from pathlib import Path

def download_vitessce():
    # Create static directory
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # URLs for Vitessce files
    js_url = "https://unpkg.com/vitessce@3.4.5/dist/umd/vitessce.min.js"
    css_url = "https://unpkg.com/vitessce@3.4.5/dist/umd/vitessce.min.css"
    
    # Download JS file
    print("Downloading Vitessce JS...")
    js_response = requests.get(js_url)
    if js_response.status_code == 200:
        with open(static_dir / "vitessce.js", "wb") as f:
            f.write(js_response.content)
        print("Successfully downloaded vitessce.js")
    else:
        print(f"Failed to download JS file: {js_response.status_code}")
    
    # Download CSS file
    print("Downloading Vitessce CSS...")
    css_response = requests.get(css_url)
    if css_response.status_code == 200:
        with open(static_dir / "vitessce.css", "wb") as f:
            f.write(css_response.content)
        print("Successfully downloaded vitessce.css")
    else:
        print(f"Failed to download CSS file: {css_response.status_code}")

if __name__ == "__main__":
    download_vitessce() 