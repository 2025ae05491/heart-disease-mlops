from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Create directory if it doesn't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

df.to_csv(
    "data/raw/heart_disease.csv",
    index=False
)

print("Dataset downloaded.")