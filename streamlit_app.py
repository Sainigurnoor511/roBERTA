import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis using rBERTa model")
st.write("An app to analyse your reviews")


user_input = st.text_input("Please enter your review >>: ")

specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest")


if (specific_model(user_input)[0]['label']) == 0:
    st.write(" ")
elif (specific_model(user_input)[0]['label']) == 1 :
    st.write("# 😊 Positive")
elif (specific_model(user_input)[0]['label']) == "NEU" :
    st.write("# 😐 Neutral")
elif (specific_model(user_input)[0]['label']) == "NEG" :
    st.write("# 😠 Negative")
else :
    st.write("# ☠️ Error ☠️")