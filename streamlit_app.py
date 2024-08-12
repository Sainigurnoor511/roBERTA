import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis using roBERTa model")
st.write("An app to analyse your reviews")


user_input = st.text_input("Please enter your review >>: ")

roberta_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest")


if (roberta_model(user_input)[0]['label']) == 0:
    st.write(" ")
elif (roberta_model(user_input)[0]['label']) == "positive" :
    st.write("# 😊 Positive")
elif (roberta_model(user_input)[0]['label']) == "neutal" :
    st.write("# 😐 Neutral")
elif (roberta_model(user_input)[0]['label']) == "negative" :
    st.write("# 😠 Negative")
else :
    st.write("# ☠️ Error ☠️")
