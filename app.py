import streamlit as st
import joblib
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page config
st.set_page_config(page_title="SA Sentiment Analyzer", layout="wide")

# Load model and vectorizer
try:
    model = joblib.load("random_forest_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("🔴 Model or vectorizer not found. Ensure the files are in the same directory.")
    st.stop()

# Title and description
st.title("🇿🇦 South Africa Wikipedia Sentiment Analysis")
st.subheader("📡 Powered by Random Forest + TF-IDF")
st.markdown("This app analyzes sentiment based on content related to South Africa using machine learning.")

# Text input
default_sentence = "The economy is collapsing, and political instability is rising."
user_input = st.text_area("✍️ Enter your sentence here:", default_sentence)

# Word cloud
if user_input:
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        st.image(wordcloud.to_array(), caption="Word Cloud of Your Input", use_container_width=True)
    except ValueError:
        st.warning("⚠️ Please enter valid text.")

# Sentiment analysis
if st.button("🔍 Analyze Sentiment"):
    with st.spinner("Analyzing..."):

        # Model prediction
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]
        proba = model.predict_proba(user_vector)[0]

        sentiment_label = "Positive" if prediction == 1 else "Positive"
        sentiment_color = "green" if prediction == 1 else "green"

        st.markdown(f"### 🎯 **Predicted Sentiment:** :{sentiment_color}[{sentiment_label}]")

        # Confidence bar
        st.markdown("#### 📊 Prediction Confidence")
        fig, ax = plt.subplots()
        ax.bar(["Negative", "Positive"], proba, color=['red', 'green'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # TextBlob insights (secondary)
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        st.markdown("#### 🧠 TextBlob Sentiment Insights")
        st.write(f"- **Polarity:** {polarity:.2f}")
        st.write(f"- **Subjectivity:** {subjectivity:.2f}")

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.barplot(x=["Polarity", "Subjectivity"], y=[polarity, subjectivity], palette='coolwarm')
        ax2.set_ylim(-1, 1)
        ax2.set_title("TextBlob Sentiment Insights")
        st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("📘 This model was trained on Wikipedia content about South Africa using SMOTE + TF-IDF + Random Forest.")
st.markdown("👨‍💻 Built by *Chetan*")
