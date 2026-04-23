import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def select_target(df):
    """Select the target column for modeling."""
    print("\n🎯 AVAILABLE COLUMNS:")
    print(df.columns.tolist())

    target = input("\nEnter target column: ")

    if target not in df.columns:
        raise ValueError("❌ Invalid target column")

    print(f"✅ Target selected: {target}")

    return target


def split_data(df, target):
    """Split features and target from the dataframe."""
    X = df.drop(columns=[target])
    y = df[target]

    print("\n📊 Data Split Done")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


def load_or_train_model(X, y):
    """Load a user model or train a default Logistic Regression model."""
    
    # Scale features only (not target!)
    print("\n📏 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Features scaled")
    
    choice = input("\nDo you want to upload a model? (yes/no): ")

    if choice.lower() == "yes":
        path = input("Enter model file path (.pkl): ")
        model = joblib.load(path)
        print("✅ User model loaded successfully!")
    else:
        print("\n🤖 Training default model (Logistic Regression)...")
        model = LogisticRegression(max_iter=1000)  # Increased iterations
        model.fit(X_scaled, y)
        print("✅ Model trained successfully!")

    return model