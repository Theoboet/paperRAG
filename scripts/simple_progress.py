"""Simple download progress checker - just shows folder size"""

import os
from pathlib import Path
import time

def get_folder_size_gb(folder_path):
    """Get folder size in GB."""
    total = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total / (1024**3)

# Cache location
cache = Path.home() / ".cache" / "huggingface" / "hub"

print("Checking HuggingFace cache...")
print(f"Location: {cache}")
print()

if not cache.exists():
    print("Cache folder doesn't exist yet - download hasn't started")
else:
    print("Looking for models...")
    found = False

    for item in cache.iterdir():
        if "qwen" in item.name.lower():
            print(f"Found Qwen: {item.name}")
            size = get_folder_size_gb(item)
            print(f"Size: {size:.2f} GB / ~15 GB expected")
            print(f"Progress: {(size/15)*100:.1f}%")
            found = True
            print()
        elif "tinyllama" in item.name.lower():
            print(f"Found TinyLlama: {item.name}")
            size = get_folder_size_gb(item)
            print(f"Size: {size:.2f} GB / ~2.2 GB expected")
            print(f"Progress: {(size/2.2)*100:.1f}%")
            found = True
            print()

    if not found:
        print("No model folders found yet")
        print()
        print("Available folders:")
        for item in cache.iterdir():
            if item.is_dir():
                size = get_folder_size_gb(item)
                print(f"  - {item.name[:50]}... ({size:.2f} GB)")

print()
print("Run this script repeatedly to see progress:")
print("python scripts/simple_progress.py")
