import streamlit as st
from predict import predict_sentiment

st.set_page_config(page_title="Amazon Sentiment Analyzer", layout="centered")

st.title("🛒 Amazon Review Sentiment Analyzer")
st.write("Enter a review and get its sentiment prediction.")

text = st.text_area("Review text", height=150)

if st.button("Predict sentiment"):
    if text.strip():
        sentiment = predict_sentiment(text)
        st.success(f"Predicted sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text.")
