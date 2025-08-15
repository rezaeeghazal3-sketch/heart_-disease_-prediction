import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ExternalValidator:
    def __init__(self, trained_models, preprocessor, features):
        """
        Initialize External Validator
        
        Args:
            trained_models: Dictionary of your trained models
            preprocessor: Your fitted preprocessor
            features: List of features used in training
        """
        self.trained_models = trained_models
        self.preprocessor = preprocessor
        self.features = features
        
    def load_external_dataset(self, file_path, target_column='HeartDisease'):
        """
        Load and preprocess external dataset
        
        Args:
            file_path: Path to external dataset
            target_column: Name of target column
        """
        # Load dataset
        external_df = pd.read_csv(file_path)
        
        print(f"External dataset shape: {external_df.shape}")
        print(f"Columns available: {list(external_df.columns)}")
        
        # Map column names if different
        column_mapping = {
            'age': 'Age',
            'cp': 'ChestPainType', 
            'chol': 'Cholesterol',
            'exang': 'ExerciseAngina',
            'slope': 'ST_Slope',
            'target': 'HeartDisease',
            'num': 'HeartDisease'  # Cleveland dataset uses 'num'
        }
        
        # Rename columns if needed
        for old_name, new_name in column_mapping.items():
            if old_name in external_df.columns:
                external_df = external_df.rename(columns={old_name: new_name})
        
        # Select only required features + target
        required_columns = self.features + [target_column]
        missing_columns = [col for col in required_columns if col not in external_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            return None, None
            
        external_df = external_df[required_columns].copy()
        
        # Handle target variable (convert to binary if needed)
        if target_column in external_df.columns:
            # Convert multi-class target to binary (0 = no disease, 1+ = disease)
            external_df[target_column] = (external_df[target_column] > 0).astype(int)
        
        # Remove missing values
        external_df = external_df.dropna()
        
        print(f"Final external dataset shape: {external_df.shape}")
        print(f"Target distribution: {external_df[target_column].value_counts()}")
        
        # Separate features and target
        X_external = external_df[self.features]
        y_external = external_df[target_column]
        
        return X_external, y_external
    
    def preprocess_external_data(self, X_external):
        """
        Apply same preprocessing as training data
        """
        # Create a copy for processing
        X_processed = X_external.copy()
        
        # Encode categorical variables using fitted encoders
        categorical_features = ['ChestPainType', 'ExerciseAngina', 'ST_Slope']
        
        for feature in categorical_features:
            if feature in X_processed.columns:
                if hasattr(self.preprocessor, 'label_encoders') and feature in self.preprocessor.label_encoders:
                    # Use fitted encoder
                    le = self.preprocessor.label_encoders[feature]
                    # Handle unseen categories
                    unique_values = X_processed[feature].unique()
                    for val in unique_values:
                        if val not in le.classes_:
                            print(f"Warning: Unseen category '{val}' in {feature}")
                            # Replace with most common class
                            X_processed[feature] = X_processed[feature].replace(val, le.classes_[0])
                    
                    X_processed[feature] = le.transform(X_processed[feature])
        
        # Scale features using fitted scaler
        if hasattr(self.preprocessor, 'scaler'):
            X_scaled = self.preprocessor.scaler.transform(X_processed)
            return X_scaled
        
        return X_processed.values
    
    def validate_models(self, X_external, y_external):
        """
        Perform external validation on all models
        """
        results = {}
        
        print("="*60)
        print("EXTERNAL VALIDATION RESULTS")
        print("="*60)
        
        for model_name, model in self.trained_models.items():
            print(f"\n{model_name}:")
            print("-" * 30)
            
            # Make predictions
            y_pred = model.predict(X_external)
            y_pred_proba = model.predict_proba(X_external)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_external, y_pred)
            precision = precision_score(y_external, y_pred)
            recall = recall_score(y_external, y_pred)
            f1 = f1_score(y_external, y_pred)
            
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_external, y_pred_proba)
            else:
                roc_auc = None
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print results
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            if roc_auc:
                print(f"ROC-AUC:   {roc_auc:.4f}")
        
        return results
    
    def compare_with_internal_results(self, external_results, internal_results):
        """
        Compare external validation with internal CV results
        """
        print("\n" + "="*60)
        print("INTERNAL vs EXTERNAL PERFORMANCE COMPARISON")
        print("="*60)
        
        comparison_df = []
        
        for model_name in external_results.keys():
            if model_name in internal_results:
                internal = internal_results[model_name]
                external = external_results[model_name]
                
                comparison_df.append({
                    'Model': model_name,
                    'Internal_Accuracy': internal.get('accuracy', 0),
                    'External_Accuracy': external['accuracy'],
                    'Internal_F1': internal.get('f1', 0),
                    'External_F1': external['f1'],
                    'Internal_ROC_AUC': internal.get('roc_auc', 0),
                    'External_ROC_AUC': external.get('roc_auc', 0),
                    'Accuracy_Drop': internal.get('accuracy', 0) - external['accuracy'],
                    'F1_Drop': internal.get('f1', 0) - external['f1']
                })
        
        comparison_df = pd.DataFrame(comparison_df)
        print(comparison_df.round(4))
        
        # Performance drop analysis
        print("\n" + "="*40)
        print("PERFORMANCE DROP ANALYSIS")
        print("="*40)
        
        avg_accuracy_drop = comparison_df['Accuracy_Drop'].mean()
        avg_f1_drop = comparison_df['F1_Drop'].mean()
        
        print(f"Average Accuracy Drop: {avg_accuracy_drop:.4f}")
        print(f"Average F1 Drop: {avg_f1_drop:.4f}")
        
        if avg_accuracy_drop < 0.05 and avg_f1_drop < 0.05:
            print("✅ EXCELLENT: Model generalizes very well")
        elif avg_accuracy_drop < 0.10 and avg_f1_drop < 0.10:
            print("✅ GOOD: Acceptable generalization")
        elif avg_accuracy_drop < 0.15 and avg_f1_drop < 0.15:
            print("⚠️  MODERATE: Some overfitting detected")
        else:
            print("❌ POOR: Significant overfitting - model may not generalize well")
        
        return comparison_df
    
    def plot_external_validation_results(self, results, y_external):
        """
        Plot external validation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('External Validation Results', fontsize=16)
        
        # 1. Model Performance Comparison
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        performance_data = []
        for metric in metrics:
            for model in models:
                performance_data.append({
                    'Model': model,
                    'Metric': metric.capitalize(),
                    'Score': results[model][metric]
                })
        
        performance_df = pd.DataFrame(performance_data)
        
        sns.barplot(data=performance_df, x='Model', y='Score', hue='Metric', ax=axes[0,0])
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. ROC Curves (if available)
        from sklearn.metrics import roc_curve
        
        for model_name, result in results.items():
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_external, result['y_pred_proba'])
                axes[0,1].plot(fpr, tpr, label=f"{model_name} (AUC: {result['roc_auc']:.3f})")
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves - External Validation')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. Confusion Matrix for Best Model
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        cm = confusion_matrix(y_external, results[best_model]['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title(f'Confusion Matrix - {best_model}')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # 4. Feature Importance (if available)
        if hasattr(list(self.trained_models.values())[0], 'feature_importances_'):
            feature_names = self.features
            importances = list(self.trained_models.values())[0].feature_importances_
            
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            sns.barplot(data=feature_imp_df, x='Importance', y='Feature', ax=axes[1,1])
            axes[1,1].set_title('Feature Importance')
        else:
            axes[1,1].text(0.5, 0.5, 'Feature importance\nnot available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage
def run_external_validation():
    """
    Example of how to run external validation
    """
    # Initialize your trained components (replace with your actual objects)
    # trained_models = your_trainer.models
    # preprocessor = your_preprocessor
    # features = ['Age', 'Cholesterol', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']
    
    # Initialize validator
    # validator = ExternalValidator(trained_models, preprocessor, features)
    
    # Load external dataset
    # X_external, y_external = validator.load_external_dataset('path/to/external/dataset.csv')
    
    # Preprocess external data
    # X_external_processed = validator.preprocess_external_data(X_external)
    
    # Validate models
    # external_results = validator.validate_models(X_external_processed, y_external)
    
    # Compare with internal results (if available)
    # internal_results = {...}  # Your CV results
    # comparison = validator.compare_with_internal_results(external_results, internal_results)
    
    # Plot results
    # validator.plot_external_validation_results(external_results, y_external)
    
    print("External validation completed!")

if __name__ == "__main__":
    run_external_validation()