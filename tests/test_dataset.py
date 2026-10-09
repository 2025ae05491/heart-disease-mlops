import pandas as pd

def test_dataset_not_empty():

    df = pd.read_csv(
        "data/processed/heart_clean.csv"
    )

    assert len(df) > 0

def test_target_column_exists():

    df = pd.read_csv(
        "data/processed/heart_clean.csv"
    )

    assert "target" in df.columns