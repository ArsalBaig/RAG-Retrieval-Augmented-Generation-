# ------------------ Importing Lib's ------------------
import streamlit as st
import torch
import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

# ------------------ Cached Resource Loading ------------------
@st.cache_resource
def load_vectorstore():
    # 1- Load & Prepare Data
    loader = TextLoader(r'C:\Users\ok\OneDrive\Documents\LLMs\RAG(Retrival Augmented Generation)\Conversational RAG\Mobiles-Dataset-_2025_.txt')
    docs = loader.load()

    # 2- Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # 3- Chunking
    chunks = text_splitter.split_documents(docs)

    # 4- Embedding & Storing in db.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Check if DB already exists & creating new vectorstore.
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )   
    return vectorstore

# 5- Loading LLM.
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",  
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY") 
    )

# ------------------ Load Cached Resources ------------------
vectorstore = load_vectorstore()
llm = load_llm()

# ------------------ Streamlit App Setup ------------------
st.set_page_config(
    page_title="Cell Phone Conversational Bot",
    layout="centered"
)
st.title("📱 Cell Phone Conversational RAG Bot")

# ------------------ Session State ------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ RAG Function ------------------
def get_response(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    relevant_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    history_text = ""
    # 'Even' rep Human text while 'Odd' reps the Bot text.
    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            history_text += (
                f"Human: {st.session_state.chat_history[i]}\n"
                f"Assistant: {st.session_state.chat_history[i+1]}\n\n"
            )

    prompt = f"""You are a helpful assistant specialized in answering questions about mobile phones based on a product database.

Previous conversation:
{history_text if history_text else "No previous conversation"}

Context from database:
{context}

Current question:
{query}"""

    response = llm.invoke(prompt)
    return response.content

# ------------------ UI ------------------
user_query = st.text_input("Ask about any mobile phone:")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Thinking..."):
            answer = get_response(user_query)

            # Save chat history in memory.
            st.session_state.chat_history.append(user_query)
            st.session_state.chat_history.append(answer)

            st.success("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question!")

# ------------------ Chat Display ------------------
if st.session_state.chat_history:
    st.subheader("📜 Conversation History")
    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🧑 You:** {st.session_state.chat_history[i]}")
                st.markdown(f"**🤖 Assistant:** {st.session_state.chat_history[i+1]}")
                st.divider()

# ------------------ Clear History Button ------------------
if st.session_state.chat_history:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
        