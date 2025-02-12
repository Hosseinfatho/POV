import vitessce
import shutil
from pathlib import Path

def setup_vitessce():
    # Find vitessce directory
    vitessce_dir = Path(vitessce.__file__).parent
    print(f"Vitessce directory: {vitessce_dir}")

    # Look for vitessce files
    js_files = list(vitessce_dir.glob("**/vitessce.min.js"))
    css_files = list(vitessce_dir.glob("**/vitessce.min.css"))

    print("Found JS files:", js_files)
    print("Found CSS files:", css_files)

    # Create static directory
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # Copy files if found
    if js_files:
        shutil.copy2(js_files[0], static_dir / "vitessce.js")
        print(f"Copied {js_files[0]} to {static_dir / 'vitessce.js'}")
    else:
        print("No Vitessce JS file found!")

    if css_files:
        shutil.copy2(css_files[0], static_dir / "vitessce.css")
        print(f"Copied {css_files[0]} to {static_dir / 'vitessce.css'}")
    else:
        print("No Vitessce CSS file found!")

if __name__ == "__main__":
    setup_vitessce() 