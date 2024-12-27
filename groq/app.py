import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Load GROQ API key
groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="tinyllama")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Title for the Streamlit app
st.title("ChatGroq Demo")

# Initialize the ChatGroq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="tinyllama"
)

# Create the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Questions: {input}
    """
)

# Create the document chain and retriever
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()  # Fixed: Ensure "vectors" is accessed correctly
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input for the prompt
user_prompt = st.text_input("Prompt Here")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    elapsed_time = time.process_time() - start
    st.write("Process Time:", elapsed_time)
    
    st.write(response['answer'])  # Ensure this correctly accesses the answer from response

    # Testing
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(st.session_state.final_documents):  # Use final documents for display
            st.write(doc.page_content)
            st.write("-------------------")
