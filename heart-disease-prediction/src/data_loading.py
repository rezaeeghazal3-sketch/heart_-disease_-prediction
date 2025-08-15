import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, data_path='data/heart.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        
    def load_data(self):
        try:
            if self.data_path.endswith('.csv'):
                self.df = pd.read_csv(self.data_path)
            elif self.data_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.data_path)
            else:
                self.df = pd.read_csv(self.data_path)
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def get_basic_info(self):
        if self.df is None:
            return None
        return {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'head': self.df.head()
        }