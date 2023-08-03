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
from langchain.callbacks import get_openai_callback
import langchain
langchain.verbose=False

def apikey():
    try:
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
    except:
        st.info("The community version has expired.")
        key = st.text_input("Enter you OpenAI API Key")
    return key

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
@st.cache_resource
def split_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)
    return chunks

@st.cache_resource
def creating_embeddings(chunks, key):

    # Converting in vectors
    embeddings = OpenAIEmbeddings(openai_api_key=key)

    base = FAISS.from_texts(chunks, embeddings)

    return base

def user_interaction(base, key):
    user_ques = st.text_input("Ask question")
    if user_ques:
        docs = base.similarity_search(user_ques)
        chain = load_qa_chain(OpenAI(openai_api_key=key), chain_type="stuff")
        with get_openai_callback() as cb:

            response = chain.run(input_documents = docs, question=user_ques)
            print(cb)

        st.write(response)
        st.subheader("Thank you. We hope you had a good experience 😀🎓✨")


def main():
    st.title("Question PDF ❔")
    key = ''
    with st.sidebar:
        st.title(f"Welcome to :red[Question PDF] ❔")

        st.write("The community version of this app is valid only until December 2023, after which you will need to use your own OpenAI API Key to continue using the service. Please note that there may be restrictions on the number of requests per day in the community version. Use it wisely, as the responses are entirely AI-based and should be used at your own risk."
                 )
    uploaded_file = file_upload()
    version = st.sidebar.selectbox("Select your choice",('Community Version','Enter own OpenAI API Key'))
    if uploaded_file:    
        if version == 'Enter own OpenAI API Key':
            key = st.sidebar.text_input("Enter your OpenAI API Key", type='password')
        else:
            key = apikey()

        if key:  
            text = extract_text(uploaded_file)
            chunks = split_chunks(text)
            try:
                base = creating_embeddings(chunks, key)
                user_interaction(base, key)
            except:
                st.error("An error occurred. Please try again.")
        else:
            st.error("Please provide valid OpenAI API Key to continue.")

            
if __name__=='__main__':
    main()
