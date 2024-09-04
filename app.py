import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io

# Set page configuration
st.set_page_config(
    page_title="Amazon Sentiment Analysis",
    page_icon=":star:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for fonts and styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")  # Ensure you have a styles.css file in your directory

# Load Amazon logo and display it centered
amazon_logo = Image.open('amazon_logo.png')  # Ensure the image file is in the same directory
st.image(amazon_logo, width=300)  # Adjust width for larger size

# Title with logo
st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>Amazon Sentiment Analysis App</h1>", unsafe_allow_html=True)

# Sidebar with app description and history
st.sidebar.header("About the App")
st.sidebar.write("""
This application uses Natural Language Processing (NLP) techniques to analyze the sentiment of Amazon product reviews. The default dataset used is `amazon_cells_labelled.csv`, and the machine learning algorithm used is the Naive Bayes algorithm. You can input individual reviews or upload a CSV file containing multiple reviews for bulk analysis.
""")

# Load or train the model
@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load('naive_bayes_pipeline.pkl')
        metrics = joblib.load('model_metrics.pkl')
    except FileNotFoundError:
        data = load_data()
        model, metrics = train_model(data)
    return model, metrics

# Load your dataset
@st.cache_data
def load_data():
    data = pd.read_csv('amazon_cells_labelled.csv', names=['review', 'sentiment'])
    return data

# Train the model
def train_model(data):
    X = data['review']
    y = data['sentiment']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Naive Bayes pipeline
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    # Save the model and metrics
    joblib.dump(model, 'naive_bayes_pipeline.pkl')
    joblib.dump(metrics, 'model_metrics.pkl')
    
    return model, metrics

# Generate word clouds
@st.cache_data
def generate_wordclouds(data):
    positive_text = ' '.join(data[data['sentiment'] == 1]['review'])
    negative_text = ' '.join(data[data['sentiment'] == 0]['review'])
    
    positive_wc = WordCloud(width=400, height=200, background_color='white').generate(positive_text)
    negative_wc = WordCloud(width=400, height=200, background_color='white').generate(negative_text)
    
    return positive_wc, negative_wc

# Main function
def main():
    model, metrics = load_or_train_model()
    data = load_data()
    positive_wc, negative_wc = generate_wordclouds(data)
    
    # Create tabs
    tabs = st.tabs(["Single Review Analysis", "Bulk Review Analysis", "Model Performance", "Word Clouds"])
    
    # Single Review Analysis Tab
    with tabs[0]:
        st.subheader("Single Review Sentiment Analysis")
        single_review = st.text_area("Enter your review:", height=150)
        if st.button("Analyze Sentiment", key='single'):
            if single_review.strip():
                prediction = model.predict([single_review])[0]
                sentiment = "Positive" if prediction == 1 else "Negative"
                emoji = "😊" if prediction == 1 else "😢"
                st.markdown(f"<h3 style='color: {'green' if prediction == 1 else 'red'};'>Predicted Sentiment: {sentiment} {emoji}</h3>", unsafe_allow_html=True)
                
                # Update sentiment distribution chart
                st.session_state.setdefault('sentiments', []).append(prediction)
                display_sentiment_distribution()
                display_charts_single()
            else:
                st.error("Please enter a review to analyze.")
    
    # Bulk Review Analysis Tab
    with tabs[1]:
        st.subheader("Bulk Review Sentiment Analysis")
        uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column:", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'review' in df.columns:
                predictions = model.predict(df['review'])
                df['Predicted Sentiment'] = ['Positive' if pred == 1 else 'Negative' for pred in predictions]
                st.write(df[['review', 'Predicted Sentiment']])
                
                # Download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='sentiment_analysis_results.csv',
                    mime='text/csv',
                )
                
                # Update sentiment distribution chart
                st.session_state.setdefault('sentiments', []).extend(predictions)
                display_charts_bulk(df)
            else:
                st.error("The uploaded CSV file must contain a 'review' column.")
    
    # Model Performance Tab
    with tabs[2]:
        st.subheader("Model Performance Metrics")
        st.write(f"**Accuracy:** {metrics['Accuracy']:.2f}")
        st.write(f"**Precision:** {metrics['Precision']:.2f}")
        st.write(f"**Recall:** {metrics['Recall']:.2f}")
        st.write(f"**F1-Score:** {metrics['F1-Score']:.2f}")
    
    # Word Clouds Tab
    with tabs[3]:
        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)
        with col1:
            st.image(positive_wc.to_array(), caption='Positive Reviews Word Cloud')
        with col2:
            st.image(negative_wc.to_array(), caption='Negative Reviews Word Cloud')
    
    # Sentiment History in Sidebar
    st.sidebar.subheader("Sentiment Analysis History")
    if 'sentiments' in st.session_state and st.session_state['sentiments']:
        sentiments = st.session_state['sentiments']
        pos_count = sentiments.count(1)
        neg_count = sentiments.count(0)
        st.sidebar.write(f"**Total Reviews Analyzed:** {len(sentiments)}")
        st.sidebar.write(f"**Positive:** {pos_count}")
        st.sidebar.write(f"**Negative:** {neg_count}")
    else:
        st.sidebar.write("No sentiments analyzed yet.")
    
    # Footer with credit and copyright
    st.markdown("""
        <hr>
        <div style='text-align: center;'>
            Developed by <strong>GROUP ONE</strong> &copy; 2024
        </div>
    """, unsafe_allow_html=True)

# Function to display sentiment distribution chart
def display_sentiment_distribution():
    sentiments = st.session_state['sentiments']
    sentiment_counts = pd.Series(sentiments).value_counts().sort_index()
    labels = ['Negative', 'Positive']
    sizes = [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#FF6F61', '#6B8E23'], startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Function to display line and bar charts for sentiment distribution in single review
def display_charts_single():
    sentiments = st.session_state['sentiments']
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    # Line Chart
    st.subheader("Sentiment Line Chart")
    fig, ax = plt.subplots()
    ax.plot(sentiment_counts.index, sentiment_counts.values, marker='o', color='blue')
    ax.set_title('Sentiment Line Chart')
    st.pyplot(fig)
    
    # Bar Chart
    st.subheader("Sentiment Bar Chart")
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette=['#FF6F61', '#6B8E23'])
    ax.set_title('Sentiment Bar Chart')
    st.pyplot(fig)

# Function to display line and bar charts for sentiment distribution in bulk review
def display_charts_bulk(df):
    sentiment_counts = df['Predicted Sentiment'].value_counts()
    
    # Line Chart
    st.subheader("Sentiment Line Chart")
    fig, ax = plt.subplots()
    ax.plot(sentiment_counts.index, sentiment_counts.values, marker='o', color='blue')
    ax.set_title('Sentiment Line Chart')
    st.pyplot(fig)
    
    # Bar Chart
    st.subheader("Sentiment Bar Chart")
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette=['#FF6F61', '#6B8E23'])
    ax.set_title('Sentiment Bar Chart')
    st.pyplot(fig)

if __name__ == "__main__":
    main()