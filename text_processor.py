
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessor:
    """Handles text preprocessing and cleaning"""
    
    def __init__(self):
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            with st.spinner("Downloading NLTK data..."):
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        return [self.clean_text(text) for text in texts]
    
    def get_text_stats(self, text):
        """Get statistics about a text"""
        words = text.split()
        return {
            'word_count': len(words),
            'char_count': len(text),
            'unique_words': len(set(words)),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
