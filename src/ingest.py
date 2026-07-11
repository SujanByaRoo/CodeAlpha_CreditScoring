import pandas as pd
import yaml
import os

# Load config
def load_config(config_path="config.yaml"):
    """
    🧠 Reads our config.yaml file and returns it as a Python dictionary.
    So instead of hardcoding paths, we just call load_config()
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    """
    🧠 Loads the raw CSV dataset from the path defined in config.yaml
    Returns a pandas DataFrame — think of it like an Excel sheet in Python
    """
    raw_path = config["data"]["raw_path"]

    # Check if file exists
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found at {raw_path}. Please add it.")

    df = pd.read_csv(raw_path)
    print(f"✅ Data loaded successfully!")
    print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\n🔍 First 5 rows:")
    print(df.head())
    print(f"\n📋 Column names:")
    print(df.columns.tolist())
    print(f"\n❓ Missing values:")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    config = load_config()
    df = load_data(config)