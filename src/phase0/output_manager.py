import json
import os
from datetime import datetime
from pathlib import Path

class OutputManager:
    """Manages and saves all pipeline outputs to a JSON file"""
    
    def __init__(self, output_folder=None):
        """
        Initialize the OutputManager
        
        Args:
            output_folder: Path where JSON file will be saved. 
                          Defaults to AGEIS root folder.
        """
        if output_folder is None:
            # Get the AGEIS root folder (parent of src)
            self.output_folder = str(Path(__file__).parent.parent.parent)
        else:
            self.output_folder = output_folder
        
        self.output_file = os.path.join(self.output_folder, "ageis_output.json")
        
        # Initialize the outputs dictionary
        self.outputs = {
            "timestamp": datetime.now().isoformat(),
            "phase_0": {},
            "phase_1": {},
            "phase_2": {},
            "summary": {}
        }
    
    # ========== PHASE 0 OUTPUTS ==========
    def add_phase0_dataset_info(self, df, target, numerical_cols, categorical_cols):
        """Store Phase 0 dataset information"""
        self.outputs["phase_0"]["dataset_info"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "target_column": target,
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        }
    
    def add_phase0_dataset_source(self, source_type, source_path=None):
        """Store Phase 0 dataset source information"""
        self.outputs["phase_0"]["dataset_source"] = {
            "type": source_type,
            "path": source_path
        }
    
    def add_phase0_model_info(self, model_name, model_path):
        """Store Phase 0 model loading information"""
        self.outputs["phase_0"]["model_info"] = {
            "model_name": model_name,
            "model_path": model_path,
            "loaded": True
        }
    
    # ========== PHASE 1 OUTPUTS ==========
    def add_phase1_preprocessing(self, categorical_cols, numerical_cols, encoding_info=None):
        """Store Phase 1 preprocessing information"""
        self.outputs["phase_1"]["preprocessing"] = {
            "categorical_columns": categorical_cols,
            "numerical_columns": numerical_cols,
            "encoding_applied": encoding_info or "Label encoding for categorical features"
        }
    
    def add_phase1_predictions(self, predictions_shape, unique_predictions, prediction_info=None):
        """Store Phase 1 model predictions information"""
        self.outputs["phase_1"]["model_predictions"] = {
            "predictions_shape": predictions_shape,
            "unique_predictions": unique_predictions,
            "prediction_info": prediction_info or "Predictions generated from model"
        }
    
    # ========== PHASE 2 OUTPUTS ==========
    def add_phase2_sensitive_features(self, sensitive_features, detection_method="LLM"):
        """Store Phase 2 sensitive features detection"""
        self.outputs["phase_2"]["sensitive_features"] = {
            "detected_features": sensitive_features,
            "detection_method": detection_method
        }
    
    def add_phase2_distribution_bias(self, feature_name, distribution_data):
        """Store Phase 2 distribution bias analysis"""
        if "distribution_bias" not in self.outputs["phase_2"]:
            self.outputs["phase_2"]["distribution_bias"] = {}
        
        # Convert pandas Series to dict for JSON serialization
        if hasattr(distribution_data, 'to_dict'):
            self.outputs["phase_2"]["distribution_bias"][feature_name] = distribution_data.to_dict()
        else:
            self.outputs["phase_2"]["distribution_bias"][feature_name] = distribution_data
    
    def add_phase2_outcome_disparity(self, feature_name, disparity_table):
        """Store Phase 2 outcome disparity analysis"""
        if "outcome_disparity" not in self.outputs["phase_2"]:
            self.outputs["phase_2"]["outcome_disparity"] = {}
        
        # Convert pandas DataFrame to dict for JSON serialization
        if hasattr(disparity_table, 'to_dict'):
            self.outputs["phase_2"]["outcome_disparity"][feature_name] = disparity_table.to_dict(orient='records')
        else:
            self.outputs["phase_2"]["outcome_disparity"][feature_name] = disparity_table
    
    def add_phase2_fairness_metrics(self, metrics):
        """Store Phase 2 fairness metrics"""
        self.outputs["phase_2"]["fairness_metrics"] = metrics
    
    def add_phase2_llm_explanation(self, explanation):
        """Store Phase 2 LLM-based explanation"""
        self.outputs["phase_2"]["llm_explanation"] = explanation
    
    # ========== SUMMARY ==========
    def add_summary(self, summary_text):
        """Store pipeline summary"""
        self.outputs["summary"]["analysis_summary"] = summary_text
    
    def add_summary_status(self, status, message=""):
        """Store pipeline status"""
        self.outputs["summary"]["pipeline_status"] = {
            "status": status,
            "message": message,
            "completion_time": datetime.now().isoformat()
        }
    
    # ========== FILE OPERATIONS ==========
    def save(self):
        """Save all outputs to JSON file"""
        try:
            # Ensure output folder exists
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Convert any non-serializable objects
            outputs_to_save = self._make_serializable(self.outputs)
            
            # Write to JSON file
            with open(self.output_file, 'w') as f:
                json.dump(outputs_to_save, f, indent=4)
            
            print(f"\n✅ Outputs saved to: {self.output_file}")
            return True
        
        except Exception as e:
            print(f"❌ Error saving outputs: {e}")
            return False
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def get_output_path(self):
        """Get the full path to the output JSON file"""
        return self.output_file
    
    def get_outputs(self):
        """Get all collected outputs"""
        return self.outputs
