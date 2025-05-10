import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up the page
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection using NLP")
st.markdown("🚀 Enter a news article below to check if it's **Fake** or **Real** using a Machine Learning model.")

# Load the model and vectorizer
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.error("❌ Model files not found. Please make sure 'model.pkl' and 'vectorizer.pkl' are in the same folder as this app.")
    st.stop()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Input field
user_input = st.text_area("✍️ Paste the news article text here:", height=200)

# Prediction
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction.upper() == "FAKE":
            st.error("🛑 This news is likely **FAKE**.")
        else:
            st.success("✅ This news is likely **REAL**.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Inhouse 2025 Project")
