import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_dataset(df, meta):
    print("\n⚙️ PREPROCESSING STARTED\n")

    df = df.copy()

    categorical_cols = meta["categorical_cols"]
    numerical_cols = meta["numerical_cols"]

    # 🔹 1. Handle Missing Values

    print("⚠️ Handling Missing Values...")

    # Numerical → fill with mean
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"{col}: filled with mean ({mean_val:.2f})")

    # Categorical → fill with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"{col}: filled with mode ({mode_val})")

    # 🔹 2. Encode Categorical Variables

    print("\n🔤 Encoding Categorical Columns...")

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"{col}: encoded")

    print("\n✅ Preprocessing Completed!\n")

    return df, encoders