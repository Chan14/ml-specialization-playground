import sys
from pathlib import Path

import pandas as pd

# print(f"Pandas version: {pd.__version__}")
# print(f"Python interpreter path: {sys.executable}")

file_path = Path(__file__).resolve().parent / "data" / "ames_housing_subset.csv"
print(file_path.absolute)

data_dir = Path(__file__).resolve().parent / "data"
file_path = Path(data_dir) / "ames_housing_subset.csv"
print(file_path.absolute)
