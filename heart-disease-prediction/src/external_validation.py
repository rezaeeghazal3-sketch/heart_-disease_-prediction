import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

class HeartDiseasePreprocessor:
    """Heart Disease Data Preprocessor - 5 Features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
    
    def fit_transform(self, X, y=None):
        """Fit preprocessor and transform data"""
        X_processed = X.copy()
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Encode categorical features
        categorical_features = ['ChestPainType', 'ExerciseAngina', 'ST_Slope']
        
        for feature in categorical_features:
            if feature in X_processed.columns:
                le = LabelEncoder()
                X_processed[feature] = le.fit_transform(X_processed[feature].astype(str))
                self.label_encoders[feature] = le
                print(f"Encoded {feature}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Standardize all features
        X_scaled = self.scaler.fit_transform(X_processed)
        return X_scaled
    
    def transform(self, X):
        """Transform new data"""
        X_processed = X.copy()
        
        # Encode categorical features
        for feature, le in self.label_encoders.items():
            if feature in X_processed.columns:
                # Handle unseen values
                unique_values = X_processed[feature].unique()
                for val in unique_values:
                    if val not in le.classes_:
                        print(f"Warning: Unseen value '{val}' in {feature} - replacing with {le.classes_[0]}")
                        X_processed[feature] = X_processed[feature].replace(val, le.classes_[0])
                
                X_processed[feature] = le.transform(X_processed[feature].astype(str))
        
        # Standardize
        X_scaled = self.scaler.transform(X_processed)
        return X_scaled

class HeartDiseaseTrainer:
    """Heart Disease Model Trainer"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.cv_results = {}
    
    def train_and_evaluate(self, X_train, y_train):
        """Train and evaluate models"""
        print("=" * 60)
        print("Internal Model Training and Evaluation (Cross-Validation)")
        print("=" * 60)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\n🔄 {name}:")
            print("-" * 30)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = {
                'accuracy': cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'),
                'precision': cross_val_score(model, X_train, y_train, cv=cv, scoring='precision'),
                'recall': cross_val_score(model, X_train, y_train, cv=cv, scoring='recall'),
                'f1': cross_val_score(model, X_train, y_train, cv=cv, scoring='f1'),
                'roc_auc': cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            }
            
            # Calculate means
            results = {}
            for metric, scores in cv_scores.items():
                mean_score = scores.mean()
                std_score = scores.std()
                results[metric] = mean_score
                print(f"{metric.capitalize()}: {mean_score:.4f} (±{std_score:.4f})")
            
            self.cv_results[name] = results
        
        return self.cv_results

class ExternalValidator:
    """External Validation"""
    
    def __init__(self, trained_models, preprocessor, features):
        self.trained_models = trained_models
        self.preprocessor = preprocessor
        self.features = features
    
    def load_external_dataset(self, base_path='Heart Disease data'):
        """Load external dataset from desktop path"""
        try:
            # Define the specific path structure
            desktop_path = os.path.expanduser("~/Desktop")
            project_path = os.path.join(desktop_path, "heart-disease-prediction", "data")
            
            # Try different file extensions and paths
            file_candidates = [
                os.path.join(project_path, f"{base_path}.xlsx"),
                os.path.join(project_path, f"{base_path}.xls"),
                os.path.join(project_path, f"{base_path}.csv"),
                os.path.join(desktop_path, "heart-disease-prediction", f"{base_path}.xlsx"),
                os.path.join(desktop_path, "heart-disease-prediction", f"{base_path}.xls"),
                os.path.join(desktop_path, "heart-disease-prediction", f"{base_path}.csv"),
                # Also try current directory
                f"{base_path}.xlsx",
                f"{base_path}.xls",
                f"{base_path}.csv"
            ]
            
            external_df = None
            loaded_path = None
            
            print(f"🔄 Looking for external data file...")
            print(f"📂 Checking paths:")
            for path in file_candidates[:3]:  # Show first 3 paths
                print(f"   - {path}")
            
            for file_path in file_candidates:
                try:
                    if file_path.endswith('.csv'):
                        external_df = pd.read_csv(file_path)
                    else:
                        external_df = pd.read_excel(file_path)
                    loaded_path = file_path
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"⚠️  Error loading {file_path}: {e}")
                    continue
            
            if external_df is None:
                print(f"❌ File not found in any of the expected locations:")
                for path in file_candidates:
                    print(f"   - {path}")
                print("💡 Creating sample data for testing...")
                return self.create_sample_external_data()
            
            print(f"✅ Successfully loaded from: {loaded_path}")
            print(f"📊 Initial shape: {external_df.shape}")
            print(f"📋 Available columns: {list(external_df.columns)}")
            
            # Check required columns
            required_columns = self.features + ['HeartDisease']
            missing_columns = [col for col in required_columns if col not in external_df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                print(f"📋 Available columns: {list(external_df.columns)}")
                return None, None
            
            # Select required columns
            external_df = external_df[required_columns].copy()
            
            # Data cleaning
            print("🧹 Cleaning data...")
            
            # Remove rows with missing values
            initial_shape = external_df.shape[0]
            external_df = external_df.dropna()
            final_shape = external_df.shape[0]
            
            if initial_shape != final_shape:
                print(f"⚠️  Removed {initial_shape - final_shape} rows with missing values")
            
            # Clean and validate data types
            # Age should be numeric
            if 'Age' in external_df.columns:
                external_df['Age'] = pd.to_numeric(external_df['Age'], errors='coerce')
                external_df = external_df.dropna(subset=['Age'])
                external_df['Age'] = external_df['Age'].astype(int)
            
            # Cholesterol should be numeric
            if 'Cholesterol' in external_df.columns:
                external_df['Cholesterol'] = pd.to_numeric(external_df['Cholesterol'], errors='coerce')
                external_df = external_df.dropna(subset=['Cholesterol'])
                external_df['Cholesterol'] = external_df['Cholesterol'].astype(int)
            
            # Heart Disease should be 0 or 1
            external_df['HeartDisease'] = external_df['HeartDisease'].astype(int)
            
            print(f"📊 Final cleaned shape: {external_df.shape}")
            print(f"📈 Target distribution:")
            print(external_df['HeartDisease'].value_counts())
            
            if 'Age' in external_df.columns:
                print(f"📊 Age statistics:")
                print(f"   Mean: {external_df['Age'].mean():.1f}")
                print(f"   Range: {external_df['Age'].min()}-{external_df['Age'].max()}")
            
            if 'Cholesterol' in external_df.columns:
                print(f"📊 Cholesterol statistics:")
                print(f"   Mean: {external_df['Cholesterol'].mean():.1f}")
                print(f"   Range: {external_df['Cholesterol'].min()}-{external_df['Cholesterol'].max()}")
            
            # Print categorical distributions
            categorical_features = ['ChestPainType', 'ExerciseAngina', 'ST_Slope']
            for feature in categorical_features:
                if feature in external_df.columns:
                    print(f"📊 {feature} distribution:")
                    print(external_df[feature].value_counts())
            
            # Separate features and target
            X_external = external_df[self.features]
            y_external = external_df['HeartDisease']
            
            return X_external, y_external
            
        except Exception as e:
            print(f"❌ Unexpected error loading file: {e}")
            print("💡 Creating sample data for testing...")
            return self.create_sample_external_data()
    
    def create_sample_external_data(self):
        """Create sample external data for testing"""
        print("🔧 Creating sample external data for testing...")
        
        np.random.seed(123)  # For reproducibility
        n_samples = 200
        
        external_data = {
            'Age': np.random.normal(55, 10, n_samples).astype(int),
            'Cholesterol': np.random.normal(240, 50, n_samples).astype(int),
            'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples),
            'ExerciseAngina': np.random.choice(['Y', 'N'], n_samples),
            'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples)
        }
        
        # Create target with realistic logic
        risk_factors = (
            (external_data['Age'] > 60) * 0.4 +
            (external_data['Cholesterol'] > 250) * 0.3 +
            np.array([1 if x == 'ASY' else 0 for x in external_data['ChestPainType']]) * 0.5 +
            np.array([1 if x == 'Y' else 0 for x in external_data['ExerciseAngina']]) * 0.4 +
            np.array([1 if x == 'Flat' else 0 for x in external_data['ST_Slope']]) * 0.3 +
            np.random.normal(0, 0.15, n_samples)
        )
        
        external_data['HeartDisease'] = (risk_factors > 0.5).astype(int)
        external_df = pd.DataFrame(external_data)
        
        print(f"✅ Sample data shape: {external_df.shape}")
        print(f"📈 Target distribution: \n{external_df['HeartDisease'].value_counts()}")
        
        # Separate features and target
        X_external = external_df[self.features]
        y_external = external_df['HeartDisease']
        
        return X_external, y_external
    
    def validate_models(self, X_external, y_external):
        """External validation of models"""
        if X_external is None or y_external is None:
            print("❌ No external data available")
            return {}
        
        # Preprocess external data
        print("🔄 Preprocessing external data...")
        try:
            X_external_processed = self.preprocessor.transform(X_external)
        except Exception as e:
            print(f"❌ Error preprocessing external data: {e}")
            return {}
        
        results = {}
        print("\n" + "=" * 60)
        print("🎯 External Validation Results")
        print("=" * 60)
        
        for model_name, model in self.trained_models.items():
            print(f"\n🔍 {model_name}:")
            print("-" * 30)
            
            try:
                # Predictions
                y_pred = model.predict(X_external_processed)
                y_pred_proba = model.predict_proba(X_external_processed)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_external, y_pred)
                precision = precision_score(y_external, y_pred, zero_division=0)
                recall = recall_score(y_external, y_pred, zero_division=0)
                f1 = f1_score(y_external, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_external, y_pred_proba) if y_pred_proba is not None else None
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # Display results
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                if roc_auc:
                    print(f"ROC-AUC: {roc_auc:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue
        
        return results
    
    def compare_with_internal_results(self, external_results, internal_results):
        """Compare internal and external results"""
        if not external_results or not internal_results:
            print("❌ Insufficient results for comparison")
            return pd.DataFrame()
        
        print("\n" + "=" * 60)
        print("📊 Internal vs External Performance Comparison")
        print("=" * 60)
        
        comparison_data = []
        for model_name in external_results.keys():
            if model_name in internal_results:
                internal = internal_results[model_name]
                external = external_results[model_name]
                
                comparison_data.append({
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
        
        if not comparison_data:
            print("❌ No models found for comparison")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📋 Comparison Table:")
        print(comparison_df.round(4))
        
        # Performance drop analysis
        print("\n" + "=" * 40)
        print("📉 Performance Drop Analysis")
        print("=" * 40)
        
        avg_accuracy_drop = comparison_df['Accuracy_Drop'].mean()
        avg_f1_drop = comparison_df['F1_Drop'].mean()
        
        print(f"Average Accuracy Drop: {avg_accuracy_drop:.4f}")
        print(f"Average F1 Drop: {avg_f1_drop:.4f}")
        
        if avg_accuracy_drop < 0.05 and avg_f1_drop < 0.05:
            print("✅ Status: EXCELLENT - Model generalizes very well")
        elif avg_accuracy_drop < 0.10 and avg_f1_drop < 0.10:
            print("✅ Status: GOOD - Acceptable generalization")
        elif avg_accuracy_drop < 0.15 and avg_f1_drop < 0.15:
            print("⚠️  Status: MODERATE - Some overfitting detected")
        else:
            print("❌ Status: POOR - Significant overfitting")
        
        return comparison_df
    
    def plot_results(self, results, y_external, internal_results=None):
        """Plot validation results"""
        if not results:
            print("❌ No results to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('External Validation Results - Heart Disease Data', fontsize=16)
        
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
        
        # 2. ROC Curves
        for model_name, result in results.items():
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_external, result['y_pred_proba'])
                axes[0,1].plot(fpr, tpr, label=f"{model_name} (AUC: {result['roc_auc']:.3f})")
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. Confusion Matrix for Best Model
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        cm = confusion_matrix(y_external, results[best_model]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title(f'Confusion Matrix - {best_model}')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # 4. Internal vs External Comparison
        if internal_results:
            comparison_data = []
            for model in models:
                if model in internal_results:
                    comparison_data.extend([
                        {'Model': model, 'Type': 'Internal (CV)', 'F1': internal_results[model]['f1']},
                        {'Model': model, 'Type': 'External', 'F1': results[model]['f1']}
                    ])
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                sns.barplot(data=comparison_df, x='Model', y='F1', hue='Type', ax=axes[1,1])
                axes[1,1].set_title('Internal vs External F1-Score')
                axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
                axes[1,1].legend()
            else:
                axes[1,1].text(0.5, 0.5, 'Internal results\nnot available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Performance Comparison')
        else:
            axes[1,1].text(0.5, 0.5, 'Internal results\nnot available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Performance Comparison')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def load_training_data():
    """Load real training data from the same path structure"""
    desktop_path = os.path.expanduser("~/Desktop")
    project_path = os.path.join(desktop_path, "heart-disease-prediction", "data")
    
    # Try to find training data file
    training_candidates = [
        os.path.join(project_path, "training_data.xlsx"),
        os.path.join(project_path, "training_data.csv"),
        os.path.join(project_path, "train.xlsx"),
        os.path.join(project_path, "train.csv"),
        # If external file can be split
        os.path.join(project_path, "Heart Disease data.xlsx"),
        os.path.join(project_path, "Heart Disease data.csv")
    ]
    
    for file_path in training_candidates:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            print(f"✅ Loaded training data from: {file_path}")
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            continue
    
    print("ℹ️  No separate training data found, will create synthetic training data")
    return create_sample_heart_dataset()

def create_sample_heart_dataset():
    """Create sample heart disease dataset - 5 features"""
    print("🔧 Creating sample training data...")
    
    np.random.seed(42)
    n_samples = 800
    
    # Features matching your dataset
    data = {
        'Age': np.random.normal(54, 9, n_samples).astype(int),
        'Cholesterol': np.random.normal(246, 52, n_samples).astype(int),
        'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples),
        'ExerciseAngina': np.random.choice(['Y', 'N'], n_samples),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples)
    }
    
    # Create target with medical logic
    risk_factors = (
        (data['Age'] > 60) * 0.35 +
        (data['Cholesterol'] > 240) * 0.25 +
        np.array([1 if x == 'ASY' else 0 for x in data['ChestPainType']]) * 0.4 +
        np.array([1 if x == 'Y' else 0 for x in data['ExerciseAngina']]) * 0.35 +
        np.array([1 if x == 'Flat' else 0 for x in data['ST_Slope']]) * 0.3 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    data['HeartDisease'] = (risk_factors > 0.4).astype(int)
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges
    df['Age'] = df['Age'].clip(30, 80)
    df['Cholesterol'] = df['Cholesterol'].clip(100, 400)
    
    print(f"✅ Training data shape: {df.shape}")
    print(f"📈 Target distribution: \n{df['HeartDisease'].value_counts()}")
    print(f"📊 Age statistics - Mean: {df['Age'].mean():.1f}, Range: {df['Age'].min()}-{df['Age'].max()}")
    
    return df

def run_complete_validation():
    """Run complete validation process"""
    print("🚀 Starting Complete Heart Disease Validation Process")
    print("📋 Features used: Age, ChestPainType, Cholesterol, ExerciseAngina, ST_Slope")
    print("=" * 70)
    
    # 1. Load training data (real or synthetic)
    train_df = load_training_data()
    
    # Define features matching your dataset
    features = ['Age', 'ChestPainType', 'Cholesterol', 'ExerciseAngina', 'ST_Slope']
    
    # Check if all required columns exist
    required_columns = features + ['HeartDisease']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    
    if missing_columns:
        print(f"❌ Missing columns in training data: {missing_columns}")
        print(f"📋 Available columns: {list(train_df.columns)}")
        return None
    
    X = train_df[features]
    y = train_df['HeartDisease']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 2. Preprocessing
    print("\n🔄 Preprocessing data...")
    preprocessor = HeartDiseasePreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # 3. Train models
    print("\n🎯 Training models...")
    trainer = HeartDiseaseTrainer()
    internal_results = trainer.train_and_evaluate(X_train_processed, y_train)
    
    # 4. External validation
    print("\n🔍 External validation...")
    validator = ExternalValidator(trainer.models, preprocessor, features)
    
    # Load external data with fixed path
    X_external, y_external = validator.load_external_dataset('Heart Disease data')
    
    if X_external is not None:
        # Validation
        external_results = validator.validate_models(X_external, y_external)
        
        if external_results:
            # 5. Compare results
            print("\n📊 Comparing results...")
            comparison = validator.compare_with_internal_results(external_results, internal_results)
            
            # 6. Plot results
            print("\n📈 Plotting results...")
            validator.plot_results(external_results, y_external, internal_results)
            
            print("\n✅ Validation process completed!")
            
            # Display final summary
            print("\n" + "=" * 60)
            print("🏆 FINAL SUMMARY")
            print("=" * 60)
            
            # Best model
            best_model_name = max(external_results.keys(), key=lambda x: external_results[x]['f1'])
            best_score = external_results[best_model_name]['f1']
            print(f"🥇 Best Model: {best_model_name}")
            print(f"🎯 External F1-Score: {best_score:.4f}")
            
            # Average performance drop
            if not comparison.empty:
                avg_drop = comparison['F1_Drop'].mean()
                print(f"📉 Average F1 Drop: {avg_drop:.4f}")
                if avg_drop < 0.05:
                    print("✅ Status: EXCELLENT - Very good generalization")
                elif avg_drop < 0.10:
                    print("✅ Status: GOOD - Acceptable generalization")
                else:
                    print("⚠️  Status: NEEDS IMPROVEMENT - Consider regularization or more diverse training data")
            
        else:
            print("❌ External validation failed")
            return None
        
        return {
            'trainer': trainer,
            'preprocessor': preprocessor,
            'validator': validator,
            'internal_results': internal_results,
            'external_results': external_results,
            'comparison': comparison
        }
    
    else:
        print("❌ External data not loaded")
        return None

if __name__ == "__main__":
    # Run complete process
    results = run_complete_validation()
    
    if results:
        print("\n🎉 Execution completed successfully!")
        print("💡 Make sure 'Heart Disease data.xlsx' is in ~/Desktop/heart-disease-prediction/data/")
    else:
        print("❌ Execution failed")