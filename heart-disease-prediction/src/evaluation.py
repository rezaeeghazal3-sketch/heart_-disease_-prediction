import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEvaluator:
    def __init__(self, models, X, y, n_splits=5, random_state=42):
        self.models = models
        # Convert X and y to numpy arrays to avoid index errors
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score
        }

    def stratified_kfold_cv(self):
        """Perform stratified k-fold cross-validation"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            fold_metrics = {metric: [] for metric in self.metrics}
            
            for train_idx, test_idx in skf.split(self.X, self.y):
                # Use direct numpy indexing
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                for metric_name, metric_fn in self.metrics.items():
                    if metric_name == 'roc_auc' and y_proba is not None:
                        score = metric_fn(y_test, y_proba)
                    else:
                        score = metric_fn(y_test, y_pred)
                    fold_metrics[metric_name].append(score)
            
            self.cv_results[model_name] = fold_metrics
        
        return self.cv_results

    def calculate_confidence_intervals(self, confidence=0.95):
        """Calculate confidence intervals for all metrics"""
        ci_results = {}
        
        for model_name, metrics in self.cv_results.items():
            model_ci = {}
            for metric_name, scores in metrics.items():
                mean_score = np.mean(scores)
                std_err = np.std(scores) / np.sqrt(len(scores))
                z_score = norm.ppf((1 + confidence) / 2)
                margin = z_score * std_err
                
                model_ci[metric_name] = {
                    'mean': mean_score,
                    'ci_lower': mean_score - margin,
                    'ci_upper': mean_score + margin,
                    'std': np.std(scores)
                }
            ci_results[model_name] = model_ci
        
        return ci_results

    def perform_mcnemar_test(self, model1_name, model2_name):
        """Perform McNemar's test between two models"""
        if model1_name not in self.models or model2_name not in self.models:
            raise ValueError("Both model names must be in the trained models")
        
        model1 = self.models[model1_name]
        model2 = self.models[model2_name]
        
        # Get predictions on the same data
        model1_pred = model1.predict(self.X)
        model2_pred = model2.predict(self.X)
        
        # Create contingency table
        a = np.sum((model1_pred == self.y) & (model2_pred == self.y))  # Both correct
        b = np.sum((model1_pred == self.y) & (model2_pred != self.y))  # Model1 correct, model2 wrong
        c = np.sum((model1_pred != self.y) & (model2_pred == self.y))  # Model1 wrong, model2 correct
        d = np.sum((model1_pred != self.y) & (model2_pred != self.y))  # Both wrong
        
        contingency_table = [[a, b], [c, d]]
        result = mcnemar(contingency_table, exact=False)
        
        return {
            'statistic': result.statistic,
            'pvalue': result.pvalue,
            'contingency_table': contingency_table
        }

    def perform_wilcoxon_test(self, model1_name, model2_name, metric='accuracy'):
        """Perform Wilcoxon signed-rank test on cross-validation results"""
        if model1_name not in self.cv_results or model2_name not in self.cv_results:
            raise ValueError("Both models must have cross-validation results")
        
        scores1 = self.cv_results[model1_name][metric]
        scores2 = self.cv_results[model2_name][metric]
        
        result = wilcoxon(scores1, scores2)
        
        return {
            'statistic': result.statistic,
            'pvalue': result.pvalue,
            'mean_diff': np.mean(scores1) - np.mean(scores2)
        }

    def print_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        if not self.cv_results:
            print("No cross-validation results available. Run stratified_kfold_cv() first.")
            return
        
        # Calculate confidence intervals
        ci_results = self.calculate_confidence_intervals()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*80)
        
        # Print cross-validation results with CIs
        print("\nSTRATIFIED K-FOLD CROSS-VALIDATION RESULTS (95% CIs):")
        for model_name, metrics in ci_results.items():
            print(f"\nModel: {model_name}")
            for metric_name, values in metrics.items():
                print(f"{metric_name.upper():<10}: {values['mean']:.4f} "
                      f"({values['ci_lower']:.4f}-{values['ci_upper']:.4f}) "
                      f"±{values['std']:.4f}")
        
        # Perform statistical tests between all model pairs
        model_names = list(self.models.keys())
        if len(model_names) > 1:
            print("\n" + "="*80)
            print("STATISTICAL SIGNIFICANCE TESTING")
            print("="*80)
            
            # McNemar's tests
            print("\nMCNEMAR'S TEST (paired predictions):")
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    result = self.perform_mcnemar_test(model1, model2)
                    print(f"\n{model1} vs {model2}:")
                    print(f"Statistic: {result['statistic']:.4f}")
                    print(f"P-value: {result['pvalue']:.4f}")
                    print("Contingency Table:")
                    print(f"Both correct: {result['contingency_table'][0][0]}")
                    print(f"{model1} correct only: {result['contingency_table'][0][1]}")
                    print(f"{model2} correct only: {result['contingency_table'][1][0]}")
                    print(f"Both wrong: {result['contingency_table'][1][1]}")
            
            # Wilcoxon tests
            print("\nWILCOXON SIGNED-RANK TEST (cross-validation scores):")
            for metric in self.metrics.keys():
                print(f"\nMetric: {metric}")
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        model1 = model_names[i]
                        model2 = model_names[j]
                        result = self.perform_wilcoxon_test(model1, model2, metric)
                        print(f"{model1} vs {model2}:")
                        print(f"Mean difference: {result['mean_diff']:.4f}")
                        print(f"Statistic: {result['statistic']:.4f}")
                        print(f"P-value: {result['pvalue']:.4f}")
