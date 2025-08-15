from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def select_features(self, features_to_keep, target_col):
        columns_to_keep = features_to_keep + [target_col]
        return self.df[columns_to_keep]
    
    def encode_categorical(self, categorical_cols):
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        return self.df
    
    def train_test_split(self, target_col, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled