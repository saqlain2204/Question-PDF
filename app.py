import os
import openai
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import langchain
langchain.verbose=False

def apikey():
    load_dotenv()
    print(os.getenv("OPENAI_API_KEY"))


def file_upload():
    uploaded_file = st.file_uploader("Choose a file", type='pdf')
    return uploaded_file

@st.cache_resource
def extract_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text=''
    for page in pdf_reader.pages:
        text = text + page.extract_text()

    return text

def split_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 100,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.split_text(text)

    return chunks

def creating_embeddings(chunks):

    # Converting in vectors
    embeddings = OpenAIEmbeddings()

    base = FAISS.from_texts(chunks, embeddings)

    return base

def user_interaction(base):
    user_ques = st.text_input("Ask question")
    if user_ques:
        docs = base.similarity_search(user_ques)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        response = chain.run(input_documents = docs, question=user_ques)

        st.write(response)


def main():
    st.title("Question PDF ❔")
    uploaded_file = file_upload()
    if uploaded_file:
        api_key=apikey()
        text = extract_text(uploaded_file)
        chunks = split_chunks(text)
        base = creating_embeddings(chunks)
        user_interaction(base)
            
if __name__=='__main__':
    main()