import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
df = pd.read_csv("data/processed/heart_clean.csv")

X = df.drop("target", axis=1)
y = df["target"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLflow experiment
mlflow.set_experiment("Heart_Disease_Prediction")

def train_and_log_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_params(model.get_params())

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        print(f"{model_name} completed")

        return {
            "model_name": model_name,
            "model_object": model,
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }

results = []
# Logistic Regression
lr_configs = [
    {"C": 0.1},
    {"C": 1},
    {"C": 10},
]

for config in lr_configs:

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        **config
    )

    result = train_and_log_model(
        model,
        f"LogReg_C_{config['C']}",
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    results.append(result)

# Random Forest
rf_configs = [
    {"n_estimators": 100},
    {"n_estimators": 200},
    {"n_estimators": 300},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

for config in rf_configs:

    model = RandomForestClassifier(
        random_state=42,
        **config
    )

    result = train_and_log_model(
        model,
         f"RF_{config['n_estimators']}_D{config.get('max_depth', 'None')}",
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(result)


results_df = pd.DataFrame(results)

metrics = [
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "roc_auc"
]

for metric in metrics:

    results_df[f"{metric}_rank"] = (
        results_df[metric]
        .rank(
            ascending=False,
            method="dense"
        )
    )

rank_columns = [
    f"{metric}_rank"
    for metric in metrics
]

results_df["total_rank"] = (
    results_df["accuracy_rank"]
    + results_df["f1_score_rank"]
    + results_df["precision_rank"]
    + results_df["recall_rank"]
    + 2 * results_df["roc_auc_rank"]
)

print(results_df)

results_df = results_df.sort_values(
    by="total_rank"
).reset_index(drop=True)

print(
    results_df[
        [
            "model_name",
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "roc_auc",
            "total_rank"
        ]
    ]
)

results_df.to_csv(
    "results/model_ranking.csv",
    index=False
)

champion = results_df.iloc[0]

print(
    f"\nChampion Model: "
    f"{champion['model_name']}"
)

print(
    f"Total Rank: "
    f"{champion['total_rank']}"
)

print("\nChampion Metrics")
print("-" * 40)

print(f"Accuracy  : {champion['accuracy']:.4f}")
print(f"F1 Score  : {champion['f1_score']:.4f}")
print(f"Precision : {champion['precision']:.4f}")
print(f"Recall    : {champion['recall']:.4f}")
print(f"ROC-AUC   : {champion['roc_auc']:.4f}")

joblib.dump(
    champion["model_object"],
    "models/final_model.pkl"
)
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print(
    f"{champion['model_name']} "
    "saved as final_model.pkl"
)

champion_summary = pd.DataFrame(
    [champion.drop("model_object")]
)

champion_summary.to_csv(
    "results/champion_model.csv",
    index=False
)

with open(
    "models/champion.txt",
    "w"
) as f:
    f.write(champion["model_name"])