# app.py
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and I’ll predict whether it’s **Positive** or **Negative**.")

review = st.text_area("✍️ Enter your movie review here:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        cleaned_review = preprocess_text(review)
        review_vec = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vec)[0]
        prob = model.predict_proba(review_vec).max()

        if prediction == 0:
            st.error(f"✅ Positive Review (Confidence: {prob:.2f})")
        else:
            st.success(f"❌ Negative Review (Confidence: {prob:.2f})")   
