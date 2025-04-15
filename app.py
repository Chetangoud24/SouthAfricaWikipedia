import streamlit as st
import joblib
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="SA Sentiment Analyzer", layout="wide")

# Load model and vectorizer
try:
    model = joblib.load("random_forest_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("🔴 Model or vectorizer file not found. Please make sure 'random_forest_sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are in the app folder.")
    st.stop()

# UI
st.title("🇿🇦 South Africa Wikipedia Sentiment Analysis")
st.subheader("📡 Powered by Random Forest Classifier")

st.markdown("""
This app analyzes sentiment based on a model trained on Wikipedia content for South Africa.
""")

default_sentence = "South Africa is known for its breathtaking landscapes and vibrant cultural heritage."
user_input = st.text_area("✍️ Enter your sentence here:", default_sentence)

# Show word cloud
if user_input:
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        st.image(wordcloud.to_array(), caption="Word Cloud of Your Input", use_container_width=True)
    except ValueError:
        st.warning("⚠️ Please enter some valid text to generate the word cloud.")

# Analyze Sentiment
if st.button("🔍 Analyze Sentiment"):
    with st.spinner("Analyzing..."):
        # TF-IDF + Model Prediction
        user_vector = vectorizer.transform([user_input])
        proba = model.predict_proba(user_vector)[0]
        prediction = proba.argmax()  # Use highest probability class

        sentiment_label = "Positive" if prediction == 1 else "Negative"
        sentiment_color = "green" if prediction == 1 else "red"

        st.markdown(f"### 🎯 **Predicted Sentiment:** :{sentiment_color}[{sentiment_label}]")

        # Show raw probabilities
        st.write(f"- **Negative Probability:** {proba[0]:.2f}")
        st.write(f"- **Positive Probability:** {proba[1]:.2f}")

        # Probability Bar Chart
        st.markdown("#### 📊 Prediction Confidence")
        fig, ax = plt.subplots()
        ax.bar(["Negative", "Positive"], proba, color=['red', 'green'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # TextBlob
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        st.markdown("#### 🧠 TextBlob Sentiment Analysis")
        st.write(f"- **Polarity:** {polarity:.2f}")
        st.write(f"- **Subjectivity:** {subjectivity:.2f}")

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.barplot(x=["Polarity", "Subjectivity"], y=[polarity, subjectivity], palette='coolwarm')
        ax2.set_ylim(-1, 1)
        ax2.set_title("TextBlob Sentiment Insights")
        st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("📘 Model trained on South Africa Wikipedia content using TextBlob + TF-IDF + SMOTE + Random Forest.")
st.markdown("👨‍💻 Created by *Chetan*")
