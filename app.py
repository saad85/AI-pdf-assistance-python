import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub



def get_text_from_pdf(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdfReader = PdfReader(pdf_doc)
        for page in pdfReader.pages:
            text +=  page.extract_text()

    return text

def get_text_chunks(raw_texts):
    splitter = CharacterTextSplitter(
                    separator='/n',
                    chunk_size=50,
                    chunk_overlap=10,
                    length_function = len
                )
    splited_text = splitter.split_text(raw_texts)
    return splited_text

def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    print('Creating embedding object')
    # embeddings = HuggingFaceInstructEmbeddings(
    # model_name="hkunlp/instructor-xl" # Allow remote execution of model code
    # )
    # print('Sucessfully created embedding object')
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # print('vectorstore ', vectorstore)
    # return vectorstore


def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain(
        llm=llm,
        memory=memory,
        retriver=vector_store.as_retriver()
    )
    return conversational_chain
    



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=':books:')
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple pdfs")
    st.text_input("Ask a question about your documents :books:")
    with st.sidebar:
        st.subheader("Your documents")
        pdf =  st.file_uploader("Upload your pdfs..", accept_multiple_files =True)
        if st.button("Process"):
            with st.spinner('Processing'):
                raw_texts = get_text_from_pdf(pdf)

                splited_text = get_text_chunks(raw_texts)
                
                vector_store = get_vector_store(splited_text)

                st.session_state.conversation = get_conversation_chain(vector_store)




if __name__ == '__main__':
    main()