import streamlit as st
import joblib

# Load trained models
fake_news_model = joblib.load("lstm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake News & Sentiment Classifier")

user_input = st.text_area("Paste a headline or tweet:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        # Transform input
        x_input = vectorizer.transform([user_input])
        
        # Fake news prediction
        preds = fake_news_model.predict(x_input)
        fake_real = "Real ✅" if preds[0] == 1 else "Fake ❌"
        
        # Sentiment prediction (simple rule-based)
        if any(word in user_input.lower() for word in ["good", "great", "happy", "love"]):
            sentiment = "Positive 😀"
        elif any(word in user_input.lower() for word in ["bad", "sad", "hate", "angry"]):
            sentiment = "Negative 😡"
        else:
            sentiment = "Neutral 😐"
        
        st.subheader("Results")
        st.write(f"**Fake News Prediction:** {fake_real}")
        st.write(f"**Sentiment:** {sentiment}")
