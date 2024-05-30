import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai

# Gizli anahtarı almak için Streamlit secrets özelliğini kullanın
openai_api_key = st.secrets["OPENAI_API_KEY"]

def download_file_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error("Failed to download file from GitHub")
        return None

# Github dosya URL'si
github_url = "https://github.com/username/repository/raw/branch/path/to/ppad_24_rag_data"

# Dosyayı indirin ve pandas ile okuyun
file_content = download_file_from_github(github_url)
if file_content:
    df = pd.read_pickle(file_content)
    st.write(df)
else:
    st.error("Could not load the data.")

def get_data(ship_name):
    filtered_reviews = df[df['ShipName'] == str(ship_name)]
    raw_text = '\n\n'.join(filtered_reviews['r_Review'])
    return raw_text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def initialize_vectorstore(chunks, force_refresh=False):
    if 'vectorstore' not in st.session_state or force_refresh:
        print("Creating new vectorstore...")
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        st.session_state['vectorstore'] = vectorstore
    else:
        print("Using cached vectorstore")

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=openai_api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with Cruise Passengers", page_icon=":ship:")
    st.write(css, unsafe_allow_html=True)
    
    # Ensure basic structure is present in session state
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = None
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None  # Explicit initialization

    st.header("Chat with Cruise Passengers :ship:")
    ship_name = st.text_input("Enter the ship name:")

    if ship_name and ship_name.strip() and ship_name in df['ShipName'].unique():
        # Check if ship name has changed or vectorstore needs initialization
        if 'current_ship' not in st.session_state or st.session_state['current_ship'] != ship_name or st.session_state['vectorstore'] is None:
            raw_text = get_data(ship_name)
            text_chunks = get_text_chunks(raw_text)
            initialize_vectorstore(text_chunks, force_refresh=True)
            st.session_state['current_ship'] = ship_name

        user_question = st.text_input("Ask a question to passengers:", key="user_question")
        
        if user_question:
            # Initialize conversation chain if needed
            if st.session_state['conversation'] is None:
                st.session_state['conversation'] = get_conversation_chain(st.session_state['vectorstore'])

            handle_userinput(user_question)
    else:
        if ship_name:  # Invalid ship name entered
            st.error("Please enter a valid ship name.")

if __name__ == '__main__':
    main()
