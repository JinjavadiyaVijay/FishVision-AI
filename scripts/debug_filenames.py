"""Quick diagnostic: compare on-disk filenames vs CSV file_name values."""
import re
from pathlib import Path

CROPS = Path(r"C:\fish_detection\OzFish\crops-zip\assets\projects\FDFML\crops")
CSV   = Path(r"C:\fish_detection\OzFish\crop_metadata.csv")

# 1. First 5 actual disk filenames
print("=== ACTUAL FILES ON DISK (first 10) ===")
disk_files = sorted(CROPS.glob("*.png"))[:10]
for f in disk_files:
    print(f"  {f.name}")

# 2. First 5 CSV file_name values
print("\n=== CSV file_name COLUMN (first 10) ===")
with open(CSV, "r") as fh:
    header = fh.readline().strip()
    print(f"  Header: {header}")
    for i in range(10):
        line = fh.readline().strip()
        parts = line.split(",")
        print(f"  uid={parts[0]}, file_name={parts[1]}")

# 3. Test suffix stripping
print("\n=== SUFFIX STRIP TEST ===")
suffix_re = re.compile(r"-\d+-\d+\.png$")
for f in disk_files[:5]:
    stripped = suffix_re.sub(".png", f.name)
    print(f"  disk: {f.name}")
    print(f"  base: {stripped}")
    print(f"  same? {f.name == stripped}")
    print()

# 4. Check if any CSV name matches any disk name
print("=== CROSS-CHECK ===")
import pandas as pd
df = pd.read_csv(CSV, dtype=str, nrows=20)
csv_names = set(df["file_name"].str.strip())
disk_names_raw = {f.name for f in disk_files}
disk_names_stripped = {suffix_re.sub(".png", f.name) for f in disk_files}

print(f"  CSV names sample:     {list(csv_names)[:3]}")
print(f"  Disk names raw:       {list(disk_names_raw)[:3]}")
print(f"  Disk names stripped:  {list(disk_names_stripped)[:3]}")
print(f"  Raw match:    {len(csv_names & disk_names_raw)}")
print(f"  Stripped match: {len(csv_names & disk_names_stripped)}")
