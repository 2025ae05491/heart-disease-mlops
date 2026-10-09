import joblib
import pandas as pd

# Load saved artifacts
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sample patient data
df = pd.read_csv("data/processed/heart_clean.csv")

sample_index = df.sample(
    n=1,
    random_state=42
).index

sample_data = df.drop("target", axis=1).loc[sample_index]

actual_target = df.loc[sample_index, "target"].iloc[0]

with open("models/champion.txt", "r") as f:
    champion_name = f.read()
    
if "LogReg" in champion_name:
    sample_data = scaler.transform(sample_data)

prediction = model.predict(sample_data)
probability = model.predict_proba(sample_data)

print(f"Champion Model: {champion_name}")

print(f"Class Probabilities: {probability[0]}")

if prediction[0] == 1:
    print("Prediction: Heart Disease Present")
else:
    print("Prediction: No Heart Disease")

if actual_target == 1:
    print("Actual: Heart Disease Present")
else:
    print("Actual: No Heart Disease")


'''
Preprocessing Pipeline and Reproducibility

Data preprocessing was standardized using StandardScaler for Logistic Regression models. The fitted scaler object was serialized and stored as scaler.pkl using Joblib. During inference, the same scaler is loaded and applied to incoming data before prediction. This ensures that feature transformations remain identical between training and deployment environments, guaranteeing reproducible model behavior.