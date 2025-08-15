from data_loading import DataLoader
from eda import EDA
from preprocessing import DataPreprocessor
from training import ModelTrainer
from evaluation import ModelEvaluator
from visualization import Visualizer
from prediction import Predictor
from recommendations import HeartRecommendationSystem

def main():
    # Initialize components
    data_loader = DataLoader('C:/Users/Ghazal/Desktop/heart-disease-prediction/data/heart.csv')

    if not data_loader.load_data():
        print("Failed to load data")
        return
    
    # Perform EDA
    eda = EDA(data_loader.df)
    print("Missing values:", eda.analyze_missing_values())
    print("Target distribution:", eda.analyze_target('HeartDisease'))
    
    # Preprocess data
    preprocessor = DataPreprocessor(data_loader.df)
    features = ['Age', 'Cholesterol', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']
    df_processed = preprocessor.select_features(features, 'HeartDisease')
    df_processed = preprocessor.encode_categorical(['ChestPainType', 'ExerciseAngina', 'ST_Slope'])
    X_train, X_test, y_train, y_test = preprocessor.train_test_split('HeartDisease')
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Comprehensive evaluation
    print("\nRunning comprehensive model evaluation...")
    evaluator = ModelEvaluator(trainer.models, X_train_scaled, y_train)
    
    # 1. Stratified K-Fold CV
    cv_results = evaluator.stratified_kfold_cv()
    
    # 2. Statistical tests and confidence intervals
    evaluator.print_evaluation_report()
    
    # Make predictions for a sample patient
    predictor = Predictor(trainer.models, preprocessor.scaler, 
                         preprocessor.label_encoders, features)
    
    sample_patient = {
        'Age': 58,
        'Cholesterol': 280,
        'ChestPainType': 'ATA',
        'ExerciseAngina': 'Y',
        'ST_Slope': 'Flat'
    }
    
    print("\nMaking prediction for sample patient:")
    print(sample_patient)
    predictions = predictor.predict_new_patient(sample_patient)
    
    print("\nPrediction Results:")
    for model_name, pred in predictions.items():
        diagnosis = "Heart Disease" if pred['prediction'] == 1 else "No Heart Disease"
        print(f"{model_name}: {diagnosis} (confidence: {pred['probability']:.2%})")
    
    # Get recommendations
    recommendation_system = HeartRecommendationSystem()
    encoded_patient = {
        'Age': sample_patient['Age'],
        'Cholesterol': sample_patient['Cholesterol'],
        'ChestPainType': 1,
        'ExerciseAngina': 1,
        'ST_Slope': 1
    }
    
    print("\n" + "="*50)
    print("Clinical Recommendations:")
    print("="*50)
    print(recommendation_system.get_comprehensive_recommendation(encoded_patient))

if __name__ == "__main__":
    main()
