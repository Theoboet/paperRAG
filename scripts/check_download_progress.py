"""
Check LLM download progress

Monitors the HuggingFace cache to show download progress.
"""

import os
import sys
import io
from pathlib import Path
import time

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def get_folder_size(folder_path):
    """Calculate total size of a folder in GB."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        pass
    return total_size / (1024 ** 3)  # Convert to GB


def find_qwen_cache():
    """Find the Qwen model cache directory."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_root.exists():
        return None

    # Look for Qwen model folders
    for item in cache_root.iterdir():
        if "qwen" in item.name.lower() and "2.5-7b" in item.name.lower():
            return item

    return None


def monitor_download(check_interval=3, max_checks=200):
    """Monitor the download progress."""

    print("=" * 70)
    print("LLM Download Progress Monitor")
    print("=" * 70)
    print()

    # Find cache directory
    print("Looking for download cache...")
    cache_dir = find_qwen_cache()

    if not cache_dir:
        print("Cache directory not found yet.")
        print(f"Checking: {Path.home() / '.cache' / 'huggingface' / 'hub'}")
        print()
        print("Waiting for download to start...")
        print()

        # Wait for cache to appear
        for i in range(30):
            time.sleep(2)
            cache_dir = find_qwen_cache()
            if cache_dir:
                break
            print(f"Still waiting... ({i*2}s)", end='\r')

        if not cache_dir:
            print()
            print("Download hasn't started yet or cache location is different.")
            print("The model might be downloading to a different location.")
            return

    print(f"Found cache: {cache_dir}")
    print()
    print("Monitoring download progress...")
    print("(Ctrl+C to stop monitoring)")
    print()
    print("-" * 70)

    prev_size = 0
    expected_size = 15.0  # ~15GB expected

    for check_num in range(max_checks):
        current_size = get_folder_size(cache_dir)

        # Calculate progress
        progress_pct = (current_size / expected_size) * 100
        downloaded_mb = current_size * 1024  # Convert to MB for display

        # Calculate speed (if we have previous measurement)
        if prev_size > 0:
            speed_mb = ((current_size - prev_size) * 1024) / check_interval
        else:
            speed_mb = 0

        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * current_size / expected_size)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        # Display
        print(f"[{bar}] {progress_pct:.1f}%", end='')
        print(f"  {current_size:.2f} GB / {expected_size:.1f} GB", end='')
        if speed_mb > 0:
            print(f"  ({speed_mb:.1f} MB/s)", end='')
        print("     ", end='\r')

        # Check if complete
        if current_size >= expected_size * 0.95:  # 95% threshold
            print()
            print()
            print("✓ Download appears complete!")
            print(f"Total downloaded: {current_size:.2f} GB")
            break

        # Check if stalled
        if check_num > 10 and current_size == prev_size:
            print()
            print()
            print("Download appears stalled (no change in 30 seconds).")
            print(f"Current size: {current_size:.2f} GB")
            print("This might be normal - large files take time to start.")

        prev_size = current_size
        time.sleep(check_interval)

    print()
    print("-" * 70)
    print()
    print("Monitor stopped.")
    print(f"Final size: {current_size:.2f} GB")
    print()


if __name__ == "__main__":
    try:
        monitor_download()
    except KeyboardInterrupt:
        print()
        print()
        print("Monitoring stopped by user.")
        print()
