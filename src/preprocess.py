import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml
import joblib
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def handle_missing_values(df):
    print("🔧 Handling missing values...")
    for col in df.columns:
        try:
            # Try converting to number first
            df[col] = pd.to_numeric(df[col], errors='ignore')
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        except:
            df[col] = df[col].fillna(df[col].mode()[0])
    print("✅ Missing values handled!")
    return df

def encode_categoricals(df):
    """
    🧠 Convert text columns to numbers using LabelEncoder.
    Example: male=1, female=0 | car=0, radio/TV=1, education=2 etc.
    """
    print("🔧 Encoding categorical columns...")
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"   ✅ Encoded: {col}")
    return df

def feature_engineering(df):
    """
    🧠 Create NEW columns from existing ones that give the model more signal.
    Example: Credit amount per month = Credit amount / Duration
    This tells the model how heavy the monthly burden is — super useful!
    """
    print("🔧 Engineering new features...")
    df["credit_per_month"] = df["Credit amount"] / df["Duration"]
    df["age_duration_ratio"] = df["Age"] / df["Duration"]
    print("✅ New features created: credit_per_month, age_duration_ratio")
    return df

def scale_features(X_train, X_test):
    """
    🧠 StandardScaler brings all numbers to same scale.
    Example: Age (20-70) and Credit amount (100-20000) are very different ranges.
    Models get confused by this. Scaling fixes it.
    We fit ONLY on train data — never on test. Why?
    Because test data should pretend it doesn't exist during training.
    """
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
    return X_train_scaled, X_test_scaled

def handle_imbalance(X_train, y_train):
    """
    🧠 SMOTE = Synthetic Minority Oversampling Technique.
    In credit data, most people are good customers (class 0).
    Very few are bad (class 1). Model gets lazy and just predicts 0 always.
    SMOTE creates FAKE but realistic bad customer examples to balance this.
    """
    print("🔧 Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE - Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled

def preprocess(config):
    # Load raw data
    df = pd.read_csv(config["data"]["raw_path"])

    # Drop unnamed index column if exists
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # Pipeline steps
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    df = feature_engineering(df)

    # 🧠 Target column — what we're trying to predict
    # We create a binary target: credit_amount > median = 1 (high risk), else 0
    df["target"] = (df["Credit amount"] > df["Credit amount"].median()).astype(int)

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )
    print(f"✅ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Scale
    X_train, X_test = scale_features(X_train, X_test)

    # Handle imbalance
    X_train, y_train = handle_imbalance(X_train, y_train)

    # Save processed data
    processed_df = pd.DataFrame(X_train)
    processed_df.to_csv(config["data"]["processed_path"], index=False)
    print("✅ Processed data saved!")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test = preprocess(config)
    print("\n🎉 Preprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")