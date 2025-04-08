import streamlit as st
import joblib
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and vectorizer for South Africa
model = joblib.load("south_africa_random_forest_model.pkl")
vectorizer = joblib.load("south_africa_tfidf_vectorizer.pkl")

# App UI
st.title("🇿🇦 South Africa Wikipedia Sentiment Analysis")
st.subheader("Powered by Random Forest Classifier")

st.markdown("""
This app analyzes sentiment based on a model trained on Wikipedia content for South Africa.
""")

default_sentence = "South Africa is known for its breathtaking landscapes and vibrant cultural heritage."
user_input = st.text_area("✍️ Enter your sentence here:", default_sentence)

if user_input:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
    st.image(wordcloud.to_array(), caption="Word Cloud of Your Input", use_container_width=True)

if st.button("🔍 Analyze Sentiment"):
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)[0]
    proba = model.predict_proba(user_vector)[0]

    sentiment_label = "Positive" if prediction == 1 else "Negative"
    sentiment_color = "green" if prediction == 1 else "red"

    st.markdown(f"### 🎯 **Predicted Sentiment:** :{sentiment_color}[{sentiment_label}]")

    st.markdown("#### 📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(["Negative", "Positive"], proba, color=['red', 'green'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

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

st.markdown("---")
st.markdown("📘 Model trained on South Africa Wikipedia content using TextBlob + TF-IDF + SMOTE + Random Forest.")
st.markdown("👨‍💻 Created by *Chetan*")
