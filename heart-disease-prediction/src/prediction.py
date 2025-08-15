import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, models, scaler, label_encoders, feature_names):
        """
        models: dict of trained ML models
        scaler: fitted scaler object (StandardScaler, MinMaxScaler, etc.)
        label_encoders: dict of fitted LabelEncoders for categorical features
        feature_names: list of column names used during training
        """
        self.models = models
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_names = feature_names  # columns used during training

    def predict_new_patient(self, patient_data):
        """
        patient_data: dict with feature names as keys
        Returns: dict of predictions for each model
        """
        # Convert input dict to DataFrame
        patient_df = pd.DataFrame([patient_data])

        # Encode categorical features safely
        for col, encoder in self.label_encoders.items():
            if col in patient_df.columns:
                patient_df[col] = patient_df[col].astype(str)
                # Replace unseen labels with the first class of the encoder
                patient_df[col] = patient_df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                patient_df[col] = encoder.transform(patient_df[col])

        # Add missing features with default 0
        for feature in self.feature_names:
            if feature not in patient_df.columns:
                patient_df[feature] = 0

        # Keep only the features used during training, in correct order
        patient_df = patient_df[self.feature_names]

        # Ensure numeric type for all columns
        patient_df = patient_df.astype(float)

        # چک کردن اینکه آیا scaler feature names داره یا نه
        if hasattr(self.scaler, 'feature_names_in_'):
            # مطمئن شویم که ترتیب feature ها مطابق با scaler است
            expected_features = self.scaler.feature_names_in_
            if list(patient_df.columns) != list(expected_features):
                print(f"Warning: Reordering features to match scaler")
                print(f"Current order: {list(patient_df.columns)}")
                print(f"Expected order: {list(expected_features)}")
                # Reorder columns to match scaler's expected input
                patient_df = patient_df.reindex(columns=expected_features)

        # Scale features
        try:
            patient_scaled = self.scaler.transform(patient_df)
        except Exception as e:
            print(f"Scaling error: {e}")
            print(f"Patient DF shape: {patient_df.shape}")
            print(f"Patient DF columns: {list(patient_df.columns)}")
            if hasattr(self.scaler, 'feature_names_in_'):
                print(f"Scaler expected features: {self.scaler.feature_names_in_}")
            raise

        # Make predictions
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(patient_scaled)[0]

                # Check if model has predict_proba
                prob = None
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(patient_scaled)[0]
                    # Always get probability for positive class (heart disease = 1)
                    prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                elif hasattr(model, 'decision_function'):
                    # برای SVM که predict_proba نداره از decision_function استفاده می‌کنیم
                    score = model.decision_function(patient_scaled)[0]
                    # تبدیل score به probability با sigmoid function
                    prob = 1 / (1 + np.exp(-score))
                else:
                    prob = 0.5  # default probability

                predictions[name] = {
                    'prediction': int(pred),
                    'probability': float(prob)
                }
            except Exception as e:
                print(f"Error making prediction with {name}: {e}")
                predictions[name] = {
                    'prediction': 0,
                    'probability': 0.0
                }

        return predictions
