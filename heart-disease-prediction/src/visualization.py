import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_target_distribution(self, target_counts):
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(['No Disease', 'Heart Disease'], target_counts.values)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}', ha='center', va='bottom')
        ax.set_title('Target Distribution')
        return fig
    
    def plot_feature_distribution(self, df, feature):
        fig, ax = plt.subplots(figsize=(8, 6))
        if df[feature].dtype == 'object':
            df[feature].value_counts().plot(kind='bar', ax=ax)
        else:
            df[feature].hist(ax=ax)
        ax.set_title(f'{feature} Distribution')
        return fig
    
    def plot_correlation_matrix(self, df):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Feature Correlation Matrix')
        return fig