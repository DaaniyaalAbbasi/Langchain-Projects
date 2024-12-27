import requests
import streamlit as st

def get_model_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
    json = {'input':{'topic' : input_text}})

    return response.json()['output']

def get_llm_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
    json = {'input':{'topic' : input_text}})

    return response.json()['output']

st.title("Langchain Demo using API")
input_text1 = st.text_input("Write a essay on")
input_text2 = st.text_input("Write a poem on")

if input_text1:
    st.write(get_model_response(input_text1))
if input_text2:
    st.write(get_llm_response(input_text2))