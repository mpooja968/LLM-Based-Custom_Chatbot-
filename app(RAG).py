import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import time  # Import the time module


# Video Link : https://drive.google.com/file/d/1VEV-2gdh2WXqOf7yIvEBAY84Y6L6k8Ec/view?usp=sharing
# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')  # Use your own GROQ API Key
if groq_api_key is None:
    st.error("GROQ_API_KEY environment variable not set.")
    st.stop()

# Function to extract text from PDF
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks                                                    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and update the chat history
def handle_userinput(user_question):
    start_time = time.time()  # Start time
    response = st.session_state.conversation({'question': user_question})
    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Calculate processing time

    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"👤 **You**: {message.content}")
        else:
            st.write(f"🤖 **Assistant**: {message.content}")

    st.write(f"ℹ️ Processing time: {processing_time:.2f} seconds")

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="PDF Genie",
                       page_icon="📂", layout="centered")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat With your PDF 📜")
    st.markdown("### Ask questions and interact with your PDF documents effortlessly!")
    user_question = st.text_input("Drop your question:", placeholder="Type here... 📝")

    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents 🗂️")
        pdf_doc = st.file_uploader("Upload your PDF here and click on 'Process' 📄", type=['pdf'])
        if pdf_doc is not None:
            if st.button("Process"):
                start_time = time.time()  # Start time
                with st.spinner("Processing... ⏳"):
                    raw_text = get_pdf_text(pdf_doc)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Processing complete! 🎉")
                end_time = time.time()  # End time
                processing_time = end_time - start_time  # Calculate processing time
                st.write(f"ℹ️ Total processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    main()
