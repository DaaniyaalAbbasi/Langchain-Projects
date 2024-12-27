from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"
)

add_routes(
    app,
    Ollama(),
    path="/ollama"
)

model= Ollama(model = "tinyllama")
llm = Ollama(model = "gemma2:2b")

prompt1 = ChatPromptTemplate.from_template("Write a essay about {topic} with 10 words")
prompt2 = ChatPromptTemplate.from_template("Write a poem about {topic} with 10 words")

add_routes(
    app,
    prompt1 | model,
    path = "/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path = "/poem"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)