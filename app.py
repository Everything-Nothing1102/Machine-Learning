
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.data_loader import DataLoader
from utils.text_processor import TextProcessor
from utils.model_trainer import ModelTrainer

# Configure page
st.set_page_config(
    page_title="Personalized News Finder",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = []

def main():
    # Add custom CSS for background and title font
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f0e8;
        background-image: url('data:image/svg+xml;base64,[base64 encoded newspaper pattern]');
        background-repeat: repeat;
        background-attachment: fixed;
    }
    
    .main-title {
        font-family: 'Times New Roman', serif;
        font-size: 3.5rem;
        font-weight: bold;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-family: 'Times New Roman', serif;
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">📰 Personalized News Finder</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered News Classification and Recommendation System</p>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Home", "Dataset Overview", "Model Training", "News Classification", "Personalized Recommendations", "Model Performance"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Dataset Overview":
        show_dataset_overview()
    elif page == "Model Training":
        show_model_training()
    elif page == "News Classification":
        show_news_classification()
    elif page == "Personalized Recommendations":
        show_personalized_recommendations()
    elif page == "Model Performance":
        show_model_performance()

def show_home():
    st.header("Welcome to Personalized News Finder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.write("""
        - **Multi-Model Classification**: Logistic Regression and KNN
        - **Text Processing**: Advanced NLP with NLTK
        - **Personalized Recommendations**: AI-driven content filtering
        - **Real-time Analysis**: Instant news article classification
        - **Performance Metrics**: NDCG and MAP score visualization
        """)
    
    with col2:
        st.subheader("🚀 Get Started")
        st.write("""
        1. **Dataset Overview**: Explore the BBC news dataset
        2. **Model Training**: Train multiple ML models
        3. **News Classification**: Classify individual articles
        4. **Recommendations**: Get personalized news suggestions
        5. **Performance**: View model comparison metrics
        """)
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Load Dataset", use_container_width=True):
            load_dataset()
    
    with col2:
        if st.button("🤖 Train Models", use_container_width=True):
            if st.session_state.data_loaded:
                train_models()
            else:
                st.error("Please load the dataset first!")
    
    with col3:
        if st.button("📊 View Performance", use_container_width=True):
            if st.session_state.models_trained:
                st.switch_page = "Model Performance"
            else:
                st.error("Please train the models first!")

# [Additional pages: show_dataset_overview, show_model_training, show_news_classification, show_personalized_recommendations, show_model_performance]
# [Helper functions: load_dataset, train_models, classify_user_article, classify_dataset_article, get_recommendations]

if __name__ == "__main__":
    main()
