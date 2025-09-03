import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load local model + tokenizer
model_path = "./model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

st.title("Fake News & Sentiment Classifier 📰")

# User input
user_input = st.text_area("Paste a headline or tweet:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        # Tokenize
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Predict fake/real
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).item()

        fake_real = "Real ✅" if preds == 1 else "Fake ❌"

        # Very simple sentiment (keywords based)
        text_lower = user_input.lower()
        if any(word in text_lower for word in ["good", "great", "happy", "love", "excellent"]):
            sentiment = "Positive 😀"
        elif any(word in text_lower for word in ["bad", "sad", "hate", "angry", "terrible"]):
            sentiment = "Negative 😡"
        else:
            sentiment = "Neutral 😐"

        # Display results
        st.subheader("Results")
        st.write(f"**Fake News Prediction:** {fake_real}")
        st.write(f"**Sentiment:** {sentiment}")

