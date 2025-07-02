
import pandas as pd
import requests
import os
import streamlit as st

class DataLoader:
    """Handles data loading and initial processing"""
    
    def __init__(self):
        self.dataset_url = "https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv"
        self.dataset_path = "dataset.csv"
    
    def download_dataset(self):
        """Download the BBC news dataset"""
        try:
            if not os.path.exists(self.dataset_path):
                st.info("Downloading BBC news dataset...")
                response = requests.get(self.dataset_url)
                response.raise_for_status()
                
                with open(self.dataset_path, 'wb') as f:
                    f.write(response.content)
                
                st.success("Dataset downloaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error downloading dataset: {str(e)}")
            return False
    
    def load_data(self):
        """Load and return the dataset"""
        if not self.download_dataset():
            raise Exception("Failed to download dataset")
        
        try:
            df = pd.read_csv(self.dataset_path, encoding='latin-1')
            
            # Basic data validation
            if df.empty:
                raise Exception("Dataset is empty")
            
            required_columns = ['news', 'type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise Exception(f"Missing required columns: {missing_columns}")
            
            # Remove any rows with missing values in critical columns
            df = df.dropna(subset=required_columns)
            
            st.info(f"Loaded {len(df)} articles from {df['type'].nunique()} categories")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def get_dataset_info(self, df):
        """Get basic information about the dataset"""
        info = {
            'total_articles': len(df),
            'categories': df['type'].value_counts().to_dict(),
            'avg_length': df['news'].str.len().mean(),
            'min_length': df['news'].str.len().min(),
            'max_length': df['news'].str.len().max()
        }
        return info
