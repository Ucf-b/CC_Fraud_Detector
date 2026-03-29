import pandas as pd
import numpy as np
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['LogAmount'] = np.log1p(df['Amount'])
    df.drop("Amount", axis=1)
    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = df.drop('Class', axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=test_size, 
                           stratify=y, random_state=random_state)

if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    # Sauvegarder
    Path("data/processed").mkdir(exist_ok=True)
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("Data prepared successfully.")