
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, average_precision_score
import streamlit as st

class ModelTrainer:
    """Handles training of multiple ML models"""
    
    def __init__(self, text_processor):
        self.text_processor = text_processor
        self.models = {}
        self.tfidf = None
        self.metrics = {}
    
    def prepare_data(self, df):
        """Prepare data for training"""
        y = df['label'].apply(lambda x: 1 if x == 0 else 0)
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def create_tfidf_features(self, X_train, X_test):
        """Create TF-IDF features"""
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_vec = self.tfidf.fit_transform(X_train)
        X_test_vec = self.tfidf.transform(X_test)
        return X_train_vec, X_test_vec
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver='liblinear'
        )
        model.fit(X_train, y_train)
        return model
    
    def train_knn(self, X_train, y_train):
        """Train K-Nearest Neighbors model"""
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine'
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a trained model"""
        predictions = model.predict_proba(X_test)[:, 1]
        true_relevance = y_test.values if hasattr(y_test, 'values') else y_test
        ndcg = ndcg_score([true_relevance], [predictions])
        map_score = average_precision_score(true_relevance, predictions)
        return {'NDCG': ndcg, 'MAP': map_score}
    
    def train_models(self, df):
        """Train all models and return results"""
        with st.spinner("Preparing data..."):
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            X_train_vec, X_test_vec = self.create_tfidf_features(X_train, X_test)
        
        models = {}
        metrics = {}
        
        with st.spinner("Training Logistic Regression..."):
            lr_model = self.train_logistic_regression(X_train_vec, y_train)
            models["Logistic Regression"] = lr_model
            metrics["Logistic Regression"] = self.evaluate_model(lr_model, "Logistic Regression", X_test_vec, y_test)
        
        with st.spinner("Training K-Nearest Neighbors..."):
            knn_model = self.train_knn(X_train_vec, y_train)
            models["KNN"] = knn_model
            metrics["KNN"] = self.evaluate_model(knn_model, "KNN", X_test_vec, y_test)
        
        return models, metrics, self.tfidf
    
    def get_feature_importance(self, model_name):
        """Get feature importance for a given model"""
        if model_name not in self.models or self.tfidf is None:
            return {}
        
        model = self.models[model_name]
        feature_names = self.tfidf.get_feature_names_out()
        
        if model_name == "Logistic Regression":
            importance = np.abs(model.coef_[0])
        else:
            return {}
        
        top_indices = np.argsort(importance)[-20:]
        top_features = {
            feature_names[i]: importance[i] 
            for i in top_indices
        }
        
        return dict(sorted(top_features.items(), key=lambda x: x[1], reverse=True))
