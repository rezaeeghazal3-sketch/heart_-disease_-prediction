import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import os
from datetime import datetime

here = os.path.dirname(__file__)
if here not in sys.path:
    sys.path.append(here)

try:
    from data_loading import DataLoader
    from eda import EDA
    from preprocessing import DataPreprocessor

    from training import ModelTrainer
    from evaluation import ModelEvaluator
    from visualization import Visualizer
    from prediction import Predictor
    from recommendations import HeartRecommendationSystem
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all the required modules are in the 'src' directory")
    sys.exit(1) 
class HeartDiseasePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.models_loaded = False  
        self.create_widgets()
        self.fade_in()
       
        self.initialize_models()
        
    def setup_window(self):
        self.root.title("Heart Disease Prediction System")
        self.root.geometry("900x700+100+100")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)
        self.root.attributes('-alpha', 0.0)
        
    def setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        self.style.configure("Title.TLabel", 
                           font=("Arial", 16, "bold"), 
                           background="#f0f0f0",
                           foreground="#2c3e50")
        self.style.configure("Header.TLabel", 
                           font=("Arial", 12, "bold"), 
                           background="#f0f0f0",
                           foreground="#34495e")
        self.style.configure("Info.TLabel", 
                           font=("Arial", 10), 
                           background="#f0f0f0",
                           foreground="#7f8c8d")
        self.style.configure("Predict.TButton", 
                           font=("Arial", 12, "bold"),
                           padding=10)
        self.style.configure("Clear.TButton", 
                           font=("Arial", 10),
                           padding=5)
    
    def fade_in(self):
        alpha = float(self.root.attributes('-alpha'))
        if alpha < 1:
            alpha = min(1.0, alpha + 0.05)
            self.root.attributes('-alpha', alpha)
            self.root.after(30, self.fade_in)
    
    def initialize_models(self):
        try:
            self.show_loading_message("Loading and training models...")
            
            data_path = 'C:/Users/Ghazal/Desktop/heart-disease-prediction/data/heart.csv'
            self.data_loader = DataLoader(data_path)
            
            if not self.data_loader.load_data():
                raise Exception("Failed to load data")
            
          
            self.preprocessor = DataPreprocessor(self.data_loader.df)
            self.features = ['Age', 'Cholesterol', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']
            df_processed = self.preprocessor.select_features(self.features, 'HeartDisease')
            df_processed = self.preprocessor.encode_categorical(['ChestPainType', 'ExerciseAngina', 'ST_Slope'])
            X_train, X_test, y_train, y_test = self.preprocessor.train_test_split('HeartDisease')
            X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
            
            self.trainer = ModelTrainer()  # 🔧 نام کلاس درست
            self.trainer.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # predictor system
            self.predictor = Predictor(self.trainer.models,
                                       self.preprocessor.scaler,
                                       self.preprocessor.label_encoders,
                                       self.features)
            self.recommendation_system = HeartRecommendationSystem()
            
            self.models_loaded = True
            self.hide_loading_message()
        except Exception as e:
            self.hide_loading_message()
            self.models_loaded = False
            messagebox.showerror("Error", f"Error loading models: {str(e)}")
    
    def show_loading_message(self, message):
        self.loading_label = ttk.Label(self.root, text=message, style="Header.TLabel")
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")
        self.root.update()
    
    def hide_loading_message(self):
        if hasattr(self, 'loading_label') and self.loading_label:
            self.loading_label.destroy()
            self.loading_label = None
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        title_label = ttk.Label(main_frame, text="Heart Disease Prediction System", style="Title.TLabel")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        input_frame = ttk.LabelFrame(main_frame, text="Patient Information", padding="10")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Patient Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(input_frame, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        ttk.Label(input_frame, text="Age (years):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.age_var = tk.StringVar()
        self.age_entry = ttk.Entry(input_frame, textvariable=self.age_var, width=30)
        self.age_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        ttk.Label(input_frame, text="Cholesterol (mg/dL):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.cholesterol_var = tk.StringVar()
        self.cholesterol_entry = ttk.Entry(input_frame, textvariable=self.cholesterol_var, width=30)
        self.cholesterol_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        ttk.Label(input_frame, text="Chest Pain Type:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.chest_pain_var = tk.StringVar()
        self.chest_pain_combo = ttk.Combobox(input_frame, textvariable=self.chest_pain_var, state="readonly", width=28)
        self.chest_pain_combo['values'] = (
            'ATA - Atypical Angina',
            'NAP - Non-Anginal Pain',
            'ASY - Asymptomatic',
            'TA - Typical Angina'
        )
        self.chest_pain_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        ttk.Label(input_frame, text="Exercise Induced Angina:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.exercise_angina_var = tk.StringVar()
        self.exercise_angina_combo = ttk.Combobox(input_frame, textvariable=self.exercise_angina_var, state="readonly", width=28)
        self.exercise_angina_combo['values'] = ('Y - Yes', 'N - No')
        self.exercise_angina_combo.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        ttk.Label(input_frame, text="ST Slope:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.st_slope_var = tk.StringVar()
        self.st_slope_combo = ttk.Combobox(input_frame, textvariable=self.st_slope_var, state="readonly", width=28)
        self.st_slope_combo['values'] = ('Up - Upsloping', 'Flat - Flat', 'Down - Downsloping')
        self.st_slope_combo.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict_disease, style="Predict.TButton")
        self.predict_button.grid(row=0, column=0, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_form, style="Clear.TButton")
        self.clear_button.grid(row=0, column=1, padx=5)
        
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=20, font=("Consolas", 10), wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def validate_inputs(self):
        errors = []
        if not self.name_var.get().strip():
            errors.append("Patient name is required")
        try:
            age = int(self.age_var.get())
            if age < 0 or age > 120:
                errors.append("Age must be between 0 and 120 years")
        except ValueError:
            errors.append("Age must be a valid integer")
        try:
            cholesterol = int(self.cholesterol_var.get())
            if cholesterol < 0 or cholesterol > 1000:
                errors.append("Cholesterol must be between 0 and 1000")
        except ValueError:
            errors.append("Cholesterol must be a valid integer")
        if not self.chest_pain_var.get():
            errors.append("Please select chest pain type")
        if not self.exercise_angina_var.get():
            errors.append("Please select exercise induced angina")
        if not self.st_slope_var.get():
            errors.append("Please select ST slope")
        return errors
    
    def extract_dropdown_values(self):
        chest_pain_map = {
            'ATA - Atypical Angina': 'ATA',
            'NAP - Non-Anginal Pain': 'NAP',
            'ASY - Asymptomatic': 'ASY',
            'TA - Typical Angina': 'TA'
        }
        exercise_angina_map = {'Y - Yes': 'Y', 'N - No': 'N'}
        st_slope_map = {'Up - Upsloping': 'Up', 'Flat - Flat': 'Flat', 'Down - Downsloping': 'Down'}
        return {
            'ChestPainType': chest_pain_map.get(self.chest_pain_var.get()),
            'ExerciseAngina': exercise_angina_map.get(self.exercise_angina_var.get()),
            'ST_Slope': st_slope_map.get(self.st_slope_var.get())
        }
    
    def predict_disease(self):
        if not self.models_loaded:
            messagebox.showerror("Error", "Models are not loaded!")
            return
        
        errors = self.validate_inputs()
        if errors:
            messagebox.showerror("Input Error", "\n".join(errors))
            return
        
        try:
            self.status_var.set("Predicting...")
            self.root.update()
            
            dropdown_values = self.extract_dropdown_values()
            sample_patient = {
                'Age': int(self.age_var.get()),
                'Cholesterol': int(self.cholesterol_var.get()),
                'ChestPainType': dropdown_values['ChestPainType'],
                'ExerciseAngina': dropdown_values['ExerciseAngina'],
                'ST_Slope': dropdown_values['ST_Slope']
            }
            
            predictions = self.predictor.predict_new_patient(sample_patient)
            
            # recommendations encoding
            encoded_patient = {
                'Age': sample_patient['Age'],
                'Cholesterol': sample_patient['Cholesterol'],
                'ChestPainType': 1 if sample_patient['ChestPainType'] == 'ATA' else 0,
                'ExerciseAngina': 1 if sample_patient['ExerciseAngina'] == 'Y' else 0,
                'ST_Slope': 1 if sample_patient['ST_Slope'] == 'Flat' else 0
            }
            
            recommendations = self.recommendation_system.get_comprehensive_recommendation(encoded_patient)
            self.display_results(sample_patient, predictions, recommendations)
            self.status_var.set("Prediction completed")
        except Exception as e:
            messagebox.showerror("Error", f"Error in prediction: {str(e)}")
            self.status_var.set("Prediction error")
    
    def display_results(self, patient_data, predictions, recommendations):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "="*80 + "\n")
        self.results_text.insert(tk.END, f"HEART DISEASE PREDICTION REPORT - {datetime.now().strftime('%Y/%m/%d %H:%M')}\n")
        self.results_text.insert(tk.END, "="*80 + "\n\n")
        self.results_text.insert(tk.END, f"Patient Name: {self.name_var.get()}\n")
        self.results_text.insert(tk.END, f"Age: {patient_data['Age']} years\n")
        self.results_text.insert(tk.END, f"Cholesterol: {patient_data['Cholesterol']} mg/dL\n")
        self.results_text.insert(tk.END, f"Chest Pain Type: {patient_data['ChestPainType']}\n")
        self.results_text.insert(tk.END, f"Exercise Induced Angina: {patient_data['ExerciseAngina']}\n")
        self.results_text.insert(tk.END, f"ST Slope: {patient_data['ST_Slope']}\n\n")
        
        self.results_text.insert(tk.END, "INDIVIDUAL MODEL PREDICTIONS:\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n")
        for model_name, pred in predictions.items():
            diagnosis = "Heart Disease Present" if pred['prediction'] == 1 else "No Heart Disease"
            confidence = pred['probability']
            self.results_text.insert(tk.END, f"{model_name}: {diagnosis} (Confidence: {confidence:.1%})\n")
        
        positive_predictions = sum(1 for pred in predictions.values() if pred['prediction'] == 1)
        total_models = len(predictions)
        self.results_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.results_text.insert(tk.END, f"MODEL CONSENSUS: {positive_predictions}/{total_models} models predict heart disease\n")
        if positive_predictions >= total_models // 2:
            self.results_text.insert(tk.END, "FINAL RESULT: HIGH PROBABILITY of Heart Disease\n")
            self.results_text.insert(tk.END, "⚠️  RECOMMENDATION: Immediate consultation with a cardiologist\n")
        else:
            self.results_text.insert(tk.END, "FINAL RESULT: LOW PROBABILITY of Heart Disease\n")
            self.results_text.insert(tk.END, "✅ RECOMMENDATION: Continue regular health monitoring\n")
        
        self.results_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.results_text.insert(tk.END, "CLINICAL RECOMMENDATIONS:\n")
        self.results_text.insert(tk.END, "="*50 + "\n")
        self.results_text.insert(tk.END, recommendations)
        
        self.results_text.insert(tk.END, "\n\n" + "⚠️" * 20 + "\n")
        self.results_text.insert(tk.END, "IMPORTANT DISCLAIMER: This system is designed for preliminary screening only.\n")
        self.results_text.insert(tk.END, "Final diagnosis and treatment must be performed by a qualified medical professional.\n")
        self.results_text.insert(tk.END, "This tool does not replace professional medical advice, diagnosis, or treatment.\n")
        self.results_text.insert(tk.END, "⚠️" * 20 + "\n")
    
    def clear_form(self):
        self.name_var.set("")
        self.age_var.set("")
        self.cholesterol_var.set("")
        self.chest_pain_var.set("")
        self.exercise_angina_var.set("")
        self.st_slope_var.set("")
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Form cleared")

def main():
    root = tk.Tk()
    app = HeartDiseasePredictionGUI(root)
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    root.mainloop()

if __name__ == "__main__":
    main()
