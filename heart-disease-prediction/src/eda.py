import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class EDA:
    def __init__(self, df):
        self.df = df
        
    def analyze_missing_values(self):
        return self.df.isnull().sum()
    
    def analyze_target(self, target_col):
        if target_col not in self.df.columns:
            return None
        return self.df[target_col].value_counts()
    
    def get_statistical_summary(self):
        return self.df.describe()