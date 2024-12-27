from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OpenAI"] = os.getenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assisstent please respomd to the queries")
        ("user", "Question:{question}")
    ]
)

st.title("Langchain Demo")
input_text = st.text_input("Search")

llm = Ollama(model = "llama3.1")
output_parser = StrOutputParser()
chain = prompt | llm |output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))