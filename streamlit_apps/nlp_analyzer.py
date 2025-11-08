import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
from collections import Counter
import plotly.express as px

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Page config
st.set_page_config(page_title="NLP Text Analyzer", layout="wide")

# Title
st.title("NLP Text Analyzer")

# Text input
text_input = st.text_area("Enter text to analyze", height=200)

if text_input:
    # Analyze button
    if st.button("Analyze Text"):
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Word Stats", "Named Entities", "Text Summary"])
        
        # Sentiment Analysis
        with tab1:
            st.subheader("Sentiment Analysis")
            blob = TextBlob(text_input)
            sentiment = blob.sentiment
            
            # Display sentiment scores
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Polarity", f"{sentiment.polarity:.2f}")
            with col2:
                st.metric("Subjectivity", f"{sentiment.subjectivity:.2f}")
            
            # Sentiment interpretation
            if sentiment.polarity > 0:
                st.success("This text has a positive sentiment")
            elif sentiment.polarity < 0:
                st.error("This text has a negative sentiment")
            else:
                st.info("This text has a neutral sentiment")
        
        # Word Statistics
        with tab2:
            st.subheader("Word Statistics")
            words = nltk.word_tokenize(text_input)
            sentences = nltk.sent_tokenize(text_input)
            
            # Display basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", len(words))
            with col2:
                st.metric("Sentence Count", len(sentences))
            with col3:
                st.metric("Average Words per Sentence", round(len(words)/len(sentences), 1))
            
            # Word frequency
            word_freq = Counter(words)
            word_freq_df = pd.DataFrame(word_freq.most_common(10), columns=['Word', 'Frequency'])
            st.bar_chart(word_freq_df.set_index('Word'))
        
        # Named Entity Recognition
        with tab3:
            st.subheader("Named Entities")
            st.warning("Named Entity Recognition requires additional setup with spaCy")
            st.info("Install spaCy and download language model to enable this feature")
        
        # Text Summary
        with tab4:
            st.subheader("Text Summary")
            # Basic extractive summarization
            sentences = nltk.sent_tokenize(text_input)
            if len(sentences) > 3:
                st.write("Key Sentences:")
                for i, sent in enumerate(sentences[:3]):
                    st.write(f"{i+1}. {sent}")
            else:
                st.write("Text is too short for summarization")
else:
    st.info("Please enter some text to analyze")