import os
import pandas as pd
import joblib
import ast
from src.phase0.loader import load_data
from src.phase0.preprocessing import preprocess_dataset
from src.phase0.inspector import BiasInspector
from src.phase0.llm_handler import LLMHandler

# NOTE: Set your GROQ_API_KEY environment variable before running this script
# export GROQ_API_KEY=your_actual_key_here

print("=" * 60)
print("🚀 AGEIS - Bias Detection & Fairness Analysis Pipeline")
print("=" * 60)

# ========== PHASE 0: DATA & MODEL CONFIGURATION ==========
print("\n📊 PHASE 0: DATA & MODEL CONFIGURATION\n")

# Step 1: Dataset Selection
print("Step 1: Select Dataset")
print("-" * 40)
print("Available datasets:")
print("  1. Adult Income Dataset (default)")
print("  2. Custom CSV file")

dataset_choice = input("\nChoose dataset (1 or 2) [default: 1]: ").strip() or "1"

if dataset_choice == "2":
    csv_path = input("Enter path to CSV file: ").strip()
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")
else:
    df = load_data()
    print("✅ Loaded Adult Income Dataset (default)")

# Step 2: Show Column Information
print("\nStep 2: Dataset Overview")
print("-" * 40)
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")

# Auto-detect numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n📈 Numerical Columns ({len(numerical_cols)}):")
for i, col in enumerate(numerical_cols, 1):
    print(f"   {i}. {col}")

print(f"\n📝 Categorical Columns ({len(categorical_cols)}):")
for i, col in enumerate(categorical_cols, 1):
    print(f"   {i}. {col}")

# Step 3: Ask for Target Column
print("\nStep 3: Select Target Column")
print("-" * 40)
print("Which column is your target variable?")

all_columns = list(df.columns)
for i, col in enumerate(all_columns, 1):
    print(f"   {i}. {col}")

target_idx = input("\nEnter column number: ").strip()
try:
    target_idx = int(target_idx) - 1
    target = all_columns[target_idx]
    print(f"✅ Target column: {target}")
except (ValueError, IndexError):
    target = "income"  # Default fallback
    print(f"⚠️ Invalid input. Using default: {target}")

# Step 4: Optional Model Selection & Loading
print("\nStep 4: Optional Model Selection")
print("-" * 40)
model_loaded = False
model_name = None

model_input = input("Enter .pkl model path (or press Enter to skip): ").strip()

if model_input:
    try:
        if os.path.exists(model_input) and model_input.endswith('.pkl'):
            model = joblib.load(model_input)
            model_name = os.path.basename(model_input)
            print(f"✅ Model loaded: {model_name}")
            model_loaded = True
        else:
            print(f"⚠️ File not found or not a .pkl file: {model_input}")
            print("⏭️ Proceeding without model...")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⏭️ Proceeding without model...")
else:
    print("⏭️  Skipped - Using default analysis")

# ========== PREPROCESSING ==========
print("\n\n⚙️ PHASE 1: PREPROCESSING\n")

meta = {
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols
}

df, encoders = preprocess_dataset(df, meta)

# ========== MODEL PREDICTIONS (if model loaded) ==========
if model_loaded:
    print("\n\n🤖 PHASE 1.5: MODEL PREDICTIONS\n")
    try:
        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target]
        X = df[feature_cols]
        
        print(f"Making predictions with {model_name}...")
        predictions = model.predict(X)
        
        # Add predictions as a new column
        df['model_predictions'] = predictions
        target = 'model_predictions'
        
        print(f"✅ Predictions generated!")
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Unique predictions: {len(set(predictions))}")
        print(f"   Will analyze bias in MODEL PREDICTIONS instead of original target")
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        print(f"⚠️ Using original target column: {target}")
        model_loaded = False

# ========== BIAS ANALYSIS ==========
print("\n\n🔍 PHASE 2: BIAS & FAIRNESS ANALYSIS\n")

print("Step 1: Auto-Detect Sensitive Features")
print("-" * 40)

# Initialize LLM for sensitive attribute detection
llm = LLMHandler()

print("🤖 LLM analyzing columns for sensitive attributes...")
try:
    # Get LLM recommendations
    llm_response = llm.detect_sensitive(categorical_cols)
    print(f"LLM Response: {llm_response}\n")
    
    # Try to extract list from LLM response
    sensitive_features = None
    if '[' in llm_response and ']' in llm_response:
        try:
            # Extract the list from the response
            start_idx = llm_response.find('[')
            end_idx = llm_response.rfind(']') + 1
            list_str = llm_response[start_idx:end_idx]
            sensitive_features = ast.literal_eval(list_str)
            
            # Filter to only include columns that exist
            sensitive_features = [col for col in sensitive_features if col in categorical_cols]
        except:
            pass
    
    # Fallback if parsing fails
    if not sensitive_features:
        sensitive_features = ["sex", "race"] if "sex" in categorical_cols or "race" in categorical_cols else categorical_cols[:2]
    
    print(f"✅ LLM Detected Sensitive Features: {sensitive_features}\n")
    
    # Option to override
    print("Would you like to modify this selection?")
    print("  1. Keep LLM selection")
    print("  2. Manually select columns")
    
    choice = input("Choice (1 or 2) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        print("\nAvailable columns:")
        for i, col in enumerate(categorical_cols, 1):
            print(f"   {i}. {col}")
        
        sensitive_input = input("Enter column numbers (e.g., 1,2): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in sensitive_input.split(',')]
            sensitive_features = [categorical_cols[i] for i in indices if 0 <= i < len(categorical_cols)]
            if not sensitive_features:
                sensitive_features = ["sex", "race"]
            print(f"✅ Updated Sensitive Features: {sensitive_features}")
        except (ValueError, IndexError):
            print(f"⚠️ Invalid input. Keeping LLM selection: {sensitive_features}")
    
except Exception as e:
    print(f"⚠️ LLM detection error: {e}")
    print("Using default sensitive features...\n")
    sensitive_features = ["sex", "race"] if ("sex" in categorical_cols or "race" in categorical_cols) else categorical_cols[:2]
    print(f"✅ Default Sensitive Features: {sensitive_features}")

# Phase 2: Bias Inspection
print("\n")
inspector = BiasInspector(df, sensitive_features, target)

inspector.distribution_bias()
inspector.outcome_disparity()
metrics = inspector.fairness_metrics()

# LLM Layer
print("\n⏳ Generating LLM-based explanation...")
explanation = llm.explain_bias(metrics)

print("\n=== LLM Explanation ===")
print(explanation)

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
print("=" * 60)