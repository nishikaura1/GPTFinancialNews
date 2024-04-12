import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationalBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = "" #raw text in PDF
    #loop, read info, concatenate
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        #creates PDF object that has pages, want to read pages
        #loop through the pages and add to text
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 1000,chunk_overlap=200, length_function=len) #chunk_size = 1000 characters, chukn_overlap = take previous 200 characters to not lose meaning of current phrase | #credit, courtesy: Alejandro AO
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name ="hkunlp/instructor-xl")
    #creating database, generate from text
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI() #can also use HuggingFace
    #memory since chatbot remembers conversation
    memory = ConversationalBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history): #loop through chat history
        #indication from streamlit
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#developing the GUI
def main():
    load_dotenv()
    st.set_page_config(page_title="Converse with Financial News")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        #initialize
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None #check that chat_history is initialized
    st.header("Converse with Financial News")
    user_question = st.text_input("Please ask any questions you may have about the latest financial news")
    if user_question:
        handle_user_input(user_question)
    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFS here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get raw content from PDF(s)
                raw_text = get_pdf_text(pdf_docs)
                #get text chunks from content from PDF
                text_chunks = get_text_chunks(raw_text) #returns list of text chunks
                st.write(text_chunks)
                # create vector store(knowledge base) from embeddings
                vectorstore = get_vector_store(text_chunks)
                #conversation chain, generate new messages in conversation, takes history and return next elem in conversaiton
                st.session_state.conversation = get_conversation_chain(vectorstore)
                #streamlit tends to 'restart' after each event and in order to prevent reinitialization, use session_state

    st.session_state.conversation
if __name__ == '__main__':
    main()


