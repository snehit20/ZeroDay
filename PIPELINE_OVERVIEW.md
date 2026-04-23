# AGEIS Pipeline Overview
## Bias Detection & Fairness Analysis Pipeline

---

## 🎯 High-Level Architecture

```
INPUT DATA → PHASE 0 → PHASE 1 → PHASE 2 → OUTPUT ARTIFACTS
             (Setup)    (Detect)   (Mitigate)  (Export)
```

---

## 📊 PHASE 0: Data & Model Configuration

**Purpose:** Load data, configure dataset, and optionally load a pre-trained model.

### Step 1: Dataset Selection
- **User Input:** Choose between:
  - `Option 1`: Adult Income Dataset (default) - loaded from local CSV files
  - `Option 2`: Custom CSV file - user-provided path
- **Output:** Raw DataFrame `df`

### Step 2: Dataset Overview
- **Actions:**
  - Count total rows and columns
  - Auto-detect numerical columns (numeric types)
  - Auto-detect categorical columns (object types)
- **Output:** Column statistics and metadata
- **Stored In:** `OutputManager.outputs["phase_0"]["dataset_info"]`

### Step 3: Target Column Selection
- **User Input:** Select target variable for fairness analysis
- **Examples:** "income", "approved", "hired", etc.
- **Default Fallback:** "income" (if invalid input)
- **Output:** `target` variable name

### Step 4: Optional Model Loading
- **User Input:** Path to pre-trained `.pkl` model (optional)
- **Actions:**
  - If provided: Load model using `joblib.load()`
  - If skipped: Set `model_loaded = False`
- **Output:** `model` object (if loaded)
- **Stored In:** `OutputManager.outputs["phase_0"]["model_info"]`

---

## 🔍 PHASE 1: Bias Detection (Built-in Analyzer)

**Purpose:** Detect and measure bias across multiple dimensions.

### Entry Point
- **Input:** DataFrame `df`, target column, optional pre-trained model
- **Process:** BiasInspector analyzes the dataset

### Step 1: Preprocessing
- **Location:** `src/phase0/preprocessing.py`
- **Actions:**
  1. **Handle Missing Values:**
     - Numerical columns → fill with mean
     - Categorical columns → fill with mode
  2. **Encode Categorical Variables:**
     - Apply LabelEncoder to all categorical columns
     - Store encoders for later use
- **Output:** Preprocessed DataFrame, encoder mappings
- **Stored In:** `OutputManager.outputs["phase_1"]["preprocessing"]`

### Step 2: Distribution Bias Analysis
- **Location:** `BiasInspector.distribution_bias()`
- **For each sensitive feature:**
  - Calculate normalized value counts (proportions)
  - Measure group imbalance ratio
  - Visualize with bar charts
- **Output:** Distribution statistics per sensitive feature
- **Stored In:** `BiasInspector.distribution_data`

### Step 3: Outcome Disparity Analysis
- **Location:** `BiasInspector.outcome_disparity()`
- **For each sensitive feature:**
  - Create crosstab of feature × target variable
  - Normalize by group to get outcome rates
  - Measure disparity (max - min outcome rate)
- **Output:** Outcome rates per group
- **Stored In:** `BiasInspector.disparity_data`

### Step 4: Fairness Metrics Calculation
- **Location:** `BiasInspector.fairness_metrics()`
- **Computed metrics (per sensitive feature):**
  - **Demographic Parity Difference** = |P(Y=1|group1) - P(Y=1|group2)|
  - **Disparate Impact Ratio** = min(outcome_rates) / max(outcome_rates)
  - **Equal Opportunity Difference**
  - **False Positive Rate Gap**
  - **False Negative Rate Gap**
- **Output:** Structured fairness report

### Optional: LLM-Enhanced Analysis
- **Tool:** `LLMHandler` (uses Groq API + Llama 3.1)
- **Functions:**
  - `detect_sensitive()`: Identify sensitive features
  - `explain_bias()`: Generate human-readable bias explanations
- **Output:** LLM-generated insights

### Output: Bias Report (Structured)
```json
{
  "distribution_bias": {
    "gender": {
      "group_proportions": {...},
      "imbalance_ratio": 1.86
    },
    "race": {...}
  },
  "outcome_bias": {
    "gender": {
      "outcome_rates": {"male": 0.45, "female": 0.28},
      "disparity": 0.17
    }
  },
  "fairness_metrics": {
    "demographic_parity_difference": 0.15,
    "disparate_impact_ratio": 0.62,
    ...
  }
}
```

---

## ⚙️ PHASE 2: Bias Mitigation Engine

**Purpose:** Automatically mitigate detected biases using multiple strategies.

### Entry Point: BiasMitigationEngine
- **Location:** `ErrorMitigation/engine.py`
- **Input:** 
  - Preprocessed DataFrame
  - Target variable
  - Sensitive features list
  - Bias report (from Phase 1)
  - Model type (logistic_regression, random_forest, xgboost)

### Step 1: Bias Classification
- **Location:** `ErrorMitigation/classifier.py`
- **Process:**
  - Parse raw bias report
  - Compare metrics against configurable thresholds
  - Generate boolean bias tags
- **Detected Bias Types:**
  - `representation_bias` (group proportions imbalanced)
  - `outcome_bias` (positive outcome rates differ)
  - `fairness_violation` (fairness metrics violated)
  - `proxy_bias` (non-sensitive features correlate with sensitive ones)
  - `intersectional_bias` (bias at intersection of multiple features)
  - `label_bias` (systematic label noise)
- **Output:** `BiasTags = {bias_type: bool, ...}`

### Step 2: Strategy Selection
- **Location:** `ErrorMitigation/selector.py`
- **Registry:** Maps each bias type → mitigation strategies
- **Example Mapping:**
  ```
  representation_bias → [reweighting, resampling, smote]
  outcome_bias → [reweighting, threshold_optimization]
  proxy_bias → [disparate_impact_remover]
  ```
- **Process:**
  - For each detected bias type, select:
    - Primary strategy (most effective)
    - Secondary strategies (alternatives)
  - Suggest combinations for multiple bias types
- **Output:** 
  ```python
  {
    "per_bias_strategies": {...},
    "combined_strategies": [...],
    "all_unique_strategies": [...]
  }
  ```

### Step 3: Generate Candidate Pipelines
- **Location:** `ErrorMitigation/generator.py`
- **Process:**
  1. Always include **baseline** (no mitigation)
  2. Add primary strategies first
  3. Add secondary strategies
  4. Add combination strategies (e.g., "dir + reweighting")
  5. Remove duplicates
  6. Limit to `max_candidates` (default: 10)
- **Output:** Ordered list of pipeline strings
  ```python
  [
    "baseline",
    "reweighting",
    "resampling",
    "reweighting + resampling",
    ...
  ]
  ```

### Step 4: Strategy Execution & Model Training
- **Location:** 
  - Strategy execution: `ErrorMitigation/strategies.py`
  - Model training: `ErrorMitigation/trainer.py`

#### 4a. Supported Mitigation Strategies

| Strategy | Method | Output |
|----------|--------|--------|
| **Reweighting** | P(y) / P(y\|group) sample weights | Weighted dataset |
| **Resampling** | Random over/undersampling | Balanced dataset |
| **SMOTE** | Synthetic Minority Oversampling | Balanced + synthetic |
| **Disparate Impact Remover** | Feature distribution repair | Repaired features |
| **Threshold Optimization** | Group-specific decision thresholds | Per-group thresholds |
| **Fairlearn Reduction** | Constrained optimization | Optimized model |

#### 4b. Training Process
- **For each candidate pipeline:**
  1. Apply selected strategy/ies to training data
  2. Train model on transformed data:
     - Logistic Regression (always available)
     - Random Forest (if available)
     - XGBoost (if available)
  3. Generate predictions on test set
  4. Store trained model + metadata

- **Output:** Dictionary of trained models
  ```python
  {
    "baseline": {model, predictions, probabilities, ...},
    "reweighting": {model, predictions, ...},
    ...
  }
  ```

### Step 5: Model Evaluation
- **Location:** `ErrorMitigation/evaluator.py`
- **For each trained model, compute:**

#### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score

#### Fairness Metrics (per group)
- Demographic Parity Difference
- Equal Opportunity Difference
- Disparate Impact Ratio
- False Positive Rate Gap
- False Negative Rate Gap

#### Group-wise Breakdown
- Accuracy per group
- FPR per group
- FNR per group

#### Intersectional Metrics
- Metrics at intersection of multiple protected attributes

- **Output:** Comprehensive evaluation results
  ```python
  {
    "baseline": {
      "performance": {accuracy: 0.85, ...},
      "fairness": {demographic_parity_diff: 0.18, ...},
      "group_metrics": {...}
    },
    "reweighting": {...}
  }
  ```

### Step 6: Strategy Ranking
- **Location:** `ErrorMitigation/ranker.py`
- **Scoring Function:**
  ```
  score = alpha × accuracy - beta × unfairness
  ```
  where:
  - `alpha` = accuracy weight (default: 0.7)
  - `beta` = fairness weight (default: 0.3)
  - `unfairness` = demographic parity difference (or other metric)

- **Process:**
  1. Calculate score for each strategy
  2. Sort by score (descending)
  3. Identify best strategy (highest score)
  4. Generate ranking table

- **Output:** 
  ```python
  {
    "best_strategy": "reweighting",
    "best_score": 0.78,
    "ranking_table": [
      {"strategy": "reweighting", "score": 0.78, ...},
      {"strategy": "baseline", "score": 0.65, ...},
      ...
    ]
  }
  ```

### Step 7: LLM-Based Reasoning & Explanation
- **Location:** `ErrorMitigation/llm_reasoner.py`
- **Tool:** Groq API (Llama 3.1)
- **Prompts Generated:**
  - "Why is this strategy best?"
  - "What trade-offs are being made?"
  - "How does this affect each protected group?"
- **Output:** Human-readable explanation

### Step 8: Output Generation
- **Debiased Dataset:**
  - Apply best mitigation strategy to full dataset
  - Export as CSV for Phase 3
  
- **Model Output:**
  - Best trained model (pickled)
  - Predictions on full dataset
  - Model metadata and performance metrics

- **Outputs Stored In:**
  - `OutputManager.outputs["phase_2"]`
  - Artifact files (CSV, JSON, PKL)

---

## 📤 PHASE 3 & 4: Artifact Export & Analysis

### Phase 3: Dataset Comparison
- Compare original vs debiased dataset
- Export fairness metrics before/after
- Generate comparison visualizations

### Phase 4: Model Explainability & Comparison
- Feature importance analysis
- SHAP-based explanations
- Model comparison tables

---

## 🔄 Entry Points & Usage Modes

### Mode 1: Interactive (`app.py`)
- Manual dataset selection
- Manual target selection
- Manual sensitive feature identification
- Step-by-step execution with user prompts

### Mode 2: Automated (`run_pipeline.py`)
- Command-line arguments for full configuration
- Synthetic or real datasets
- Batch processing mode
- Example:
  ```bash
  python run_pipeline.py --data data.csv --target approved --sensitive gender race --alpha 0.7 --beta 0.3
  ```

### Mode 3: Testing (`test_engine.py`)
- Creates synthetic biased dataset
- Runs full pipeline with demo data
- Validates engine functionality

---

## 📊 Output Files

### JSON Output (`ageis_output.json`)
Contains all pipeline outputs organized by phase:
```json
{
  "timestamp": "2024-04-23T...",
  "phase_0": {
    "dataset_info": {...},
    "dataset_source": {...},
    "model_info": {...}
  },
  "phase_1": {
    "preprocessing": {...},
    "bias_report": {...}
  },
  "phase_2": {
    "bias_classification": {...},
    "ranking": {...},
    "llm_summary": {...}
  },
  "summary": {...}
}
```

### CSV Outputs
- `original_dataset.csv` (from Phase 1)
- `debiased_dataset.csv` (from Phase 2)
- `fairness_comparison.csv` (Phase 3)

### Model Files
- `best_model.pkl` (best trained model)
- `models.pkl` (all trained models)

---

## 🔧 Configuration & Tuning

### Key Configuration Parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `alpha` | 0.7 | Accuracy weight in scoring |
| `beta` | 0.3 | Fairness weight in scoring |
| `representation_imbalance_threshold` | 0.2 | Bias detection threshold |
| `outcome_disparity_threshold` | 0.1 | Outcome bias threshold |
| `proxy_correlation_threshold` | 0.3 | Proxy bias correlation threshold |

---

## 📋 Complete Execution Flow Diagram

```
START
  ↓
[PHASE 0] DATA SETUP
  ├─ Load Dataset (Adult or Custom)
  ├─ Detect Columns (Numerical/Categorical)
  ├─ Select Target Variable
  └─ Load Optional Model
  ↓
[PHASE 1] BIAS DETECTION
  ├─ Preprocess Data (Impute, Encode)
  ├─ Calculate Distribution Bias
  ├─ Analyze Outcome Disparity
  ├─ Compute Fairness Metrics
  └─ Generate Bias Report
  ↓
[PHASE 2] BIAS MITIGATION ENGINE
  ├─ Classify Bias Types (BiasTags)
  ├─ Select Mitigation Strategies
  ├─ Generate Candidate Pipelines
  ├─ Execute Strategies & Train Models (Multi-model)
  ├─ Evaluate Performance & Fairness
  ├─ Rank Strategies (Alpha-Beta scoring)
  ├─ LLM Explanation (Groq/Llama)
  └─ Generate Best Mitigation
  ↓
[OUTPUT] ARTIFACT EXPORT
  ├─ Save Debiased Dataset
  ├─ Save Best Model
  ├─ Save Fairness Report (JSON)
  ├─ Save Ranking Table (CSV)
  └─ Generate Summary
  ↓
END
```

---

## 🎓 Key Concepts

### Sensitive Features
Features representing protected attributes (e.g., gender, race, age) where bias is measured.

### Bias Types
1. **Representation Bias**: Groups have unequal presence in dataset
2. **Outcome Bias**: Groups get different outcomes from the model
3. **Proxy Bias**: Non-sensitive features encode sensitive information
4. **Intersectional Bias**: Bias at combinations of features

### Mitigation Strategies
Techniques applied to reduce bias while maintaining model performance.

### Fairness-Accuracy Trade-off
The balance between fair predictions (across groups) and accurate predictions (overall).

---

## 📝 Summary

Your AGEIS pipeline is a **comprehensive, production-grade bias detection and mitigation system** that:

✅ **Detects** multiple types of bias across sensitive features  
✅ **Classifies** bias types automatically  
✅ **Selects** appropriate mitigation strategies  
✅ **Trains** multiple models with different strategies  
✅ **Evaluates** both fairness and performance metrics  
✅ **Ranks** strategies using configurable scoring  
✅ **Explains** using LLM-based reasoning  
✅ **Exports** debiased datasets and trained models  

The pipeline supports **interactive, automated, and testing modes** with flexible configuration and comprehensive output tracking.
