import requests
from pathlib import Path

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

# Create libs directory
libs_dir = Path("static/libs")
libs_dir.mkdir(parents=True, exist_ok=True)

# Download required files
files = {
    "react.js": "https://unpkg.com/react@17/umd/react.production.min.js",
    "react-dom.js": "https://unpkg.com/react-dom@17/umd/react-dom.production.min.js",
    "prop-types.js": "https://unpkg.com/prop-types@15.7.2/prop-types.min.js",
    "deck.gl-core.js": "https://cdn.jsdelivr.net/npm/@deck.gl/core@8.8.27/dist.min.js",
    "deck.gl-layers.js": "https://cdn.jsdelivr.net/npm/@deck.gl/layers@8.8.27/dist.min.js",
    "deck.gl-mesh-layers.js": "https://cdn.jsdelivr.net/npm/@deck.gl/mesh-layers@8.8.27/dist.min.js",
    "loaders.gl-core.js": "https://cdn.jsdelivr.net/npm/@loaders.gl/core@3.3.1/dist/dist.min.js",
    "vitessce.js": "https://unpkg.com/vitessce@3.4.5/dist/umd/vitessce.min.js",
    "vitessce.css": "https://unpkg.com/vitessce@3.4.5/dist/umd/vitessce.min.css"
}

for filename, url in files.items():
    download_file(url, libs_dir / filename) 