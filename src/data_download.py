from ucimlrepo import fetch_ucirepo
import pandas as pd

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

df.to_csv(
    "data/raw/heart_disease.csv",
    index=False
)

print("Dataset downloaded.")