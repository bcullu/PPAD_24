import streamlit as st
from htmlTemplates import css, bot_template, user_template
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI  # Remove OpenAI
# from langchain_openai import OpenAIEmbeddings  # Remove OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Add Google AI
from dotenv import load_dotenv
import os
import time
import google.generativeai as genai

load_dotenv()
# openai_api_key = os.getenv('OPENAI_API_KEY')  # Remove OpenAI
google_api_key = os.getenv('GOOGLE_API_KEY')  # Add Google API Key

# Configure Google AI
genai.configure(api_key=google_api_key)

# --- Load Data (with Error Handling) ---
try:
    df = pd.read_pickle("/Users/batuhancullu/Documents/otel_yorum_scp/Hotel_RAG/PPAD_24/veri.pkl")
except FileNotFoundError:
    st.error("Error: Could not find veri.pkl. Please make sure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Utility Functions ---

def get_data(ship_name):
    filtered_reviews = df[df['otel_adi'] == str(ship_name)]
    if filtered_reviews.empty:
        return ""
    raw_text = '\n\n'.join(filtered_reviews['body'])
    return raw_text


def get_text_chunks(raw_text):
    if not raw_text:
        return []
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
        with st.spinner("Creating vector store..."):
            try:
                # Use GoogleGenerativeAIEmbeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004", google_api_key=google_api_key) # Specify embedding model
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                st.session_state['vectorstore'] = vectorstore
                time.sleep(2)  # Simulate loading (remove in production)
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None
    else:
        st.info("Using cached vectorstore")
    return st.session_state.get('vectorstore')



def get_conversation_chain(vectorstore):
    if vectorstore is None:
        return None
    try:
        # Use ChatGoogleGenerativeAI with Gemini 1.5 Flash
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, convert_system_message_to_human=True)  # Specify LLM, and convert sys message
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            verbose=False,  # Add verbose for debugging if needed.
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error initializing conversational chain: {e}")
        return None




def handle_userinput(user_question):
    if st.session_state.conversation:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during conversation: {e}")
            # Print more detailed error information for debugging:
            st.write(e)

    else:
        st.error("Conversation chain is not initialized.")


# --- Main App Logic ---

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Customers", page_icon=":people:")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = None
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
    if 'current_ship' not in st.session_state:
        st.session_state['current_ship'] = None

    st.header("Chat with Customers :people:")

    valid_ship_names = df['otel_adi'].unique().tolist()
    ship_name = st.selectbox(
        "Select a hotel:",
        options=valid_ship_names,
        index=None,
        placeholder="Choose an option"
    )

    if ship_name:
        if st.session_state['current_ship'] != ship_name or st.session_state['vectorstore'] is None:
            raw_text = get_data(ship_name)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                vectorstore = initialize_vectorstore(text_chunks, force_refresh=True)
                if vectorstore:
                    st.session_state['current_ship'] = ship_name
                    st.session_state['conversation'] = get_conversation_chain(vectorstore)
            else:
                st.warning(f"No reviews found for {ship_name}.")
                st.session_state['current_ship'] = None
                st.session_state['conversation'] = None

        user_question = st.text_input("Ask a question to customers:", key="user_question")

        if user_question:
            if st.session_state['conversation'] is None:
                st.error("Please select a hotel with reviews to start the conversation.")
            else:
                handle_userinput(user_question)

if __name__ == '__main__':
    main()