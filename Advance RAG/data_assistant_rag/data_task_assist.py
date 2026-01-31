# ==========================================
# 1. CORE CONFIGURATION & IMPORTS
# ==========================================

import os
import pandas as pd
import zipfile
import streamlit as st
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
# ==========================================
# --- Path Configuration ---
# ==========================================
data_path = 'Advance RAG\dataset\kaggle_winning_solutions_methods.csv.zip'

# ==========================================
# --- Data Extraction and Data Transformation ---
# ==========================================
def extract_csv_from_zip(zip_path, output_directory='extracted_data'):
    # Checks if zipfile exists
    if not os.path.exists(zip_path):
        st.error(f"FATAL ERROR: The file was not found at: {os.path.abspath(zip_path)}")
        st.stop() 
    # Checks if output directory exists, if not creates it.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_directory)
    
    csv_file_name = 'kaggle_winning_solutions_methods.csv'
    csv_file_path = os.path.join(output_directory, csv_file_name)
    
    df = pd.read_csv(csv_file_path)
    df = df.dropna()

    documents = []
    for _, row in df.iterrows():
        content = f"""
        Competition: {row.get('competition_name', '')}
        Task-Type: {row.get('task_type', '')}
        Models-Used: {row.get('models_used', '')}
        Techniques: {row.get('techniques', '')}
        Description: {row.get('description', '')}
        """
        metadata = {
            'Competition': row.get('competition_name', ''),
            'Task-Type': row.get('task_type', ''),
            'Models-Used': row.get('models_used', '')
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents
# ==========================================
# --- Vector Store Construction & Embedding ---
# ==========================================

@st.cache_resource
def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory='./chroma_task_assist_db'
    )
    return vectorstore
# ==========================================
# --- LLM & Query Handling ---
# ==========================================
@st.cache_resource
def load_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        st.error("Invalid GROQ_API_KEY. Check your .env file for a key starting with 'gsk_'.")
        st.stop()
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key= os.getenv('GROQ_API_KEY'))

def clean_query(query, llm):
    prompt = f'''Clean and clarify this query for better search results in category of
    Competiton, Task-Type, Model-Used, Techniques, Description: {query}'''
    return llm.invoke(prompt).content.strip()

def rewrite_query(llm, query):
    prompt = f"Rewrite this query to be more specific for a search engine: {query}"
    return llm.invoke(prompt).content.strip()

def rerank_docs(llm, query, docs):
    ranked_docs = []
    for doc in docs:
        score_prompt = f"On a scale of 1-5, how relevant is this doc to the query? Query: {query} Doc: {doc.page_content}."
        try:
            score = llm.invoke(score_prompt).content.strip()
            ranked_docs.append((doc, int(score)))
        except:
            ranked_docs.append((doc, 0))
    return sorted(ranked_docs, key=lambda x: x[1], reverse=True) # x[0] is doc, while x[1] is it's relevance-score

# ==========================================
# --- Main Application Logic ---
# ==========================================

st.set_page_config(page_title='Data Task Assist RAG')
st.title('🚀 Data Task Assist RAG')

llm = load_llm()

@st.cache_resource
def setup_rag():
    documents = extract_csv_from_zip(data_path)
    return build_vectorstore(documents)

vectordb = setup_rag()
user_query = st.text_input('Enter your data science task query:')
submit_button = st.button('🔍 Search', type='primary', use_container_width=True)

if submit_button and user_query:
    with st.spinner("Analyzing and retrieving..."):
        refined_query = rewrite_query(llm, user_query)
        st.info(f"🔄 Refined Query: {refined_query}")
        
        retrieved_docs = vectordb.similarity_search(refined_query, k=5)
        
        if retrieved_docs:
            # Show number of documents retrieved
            st.success(f"✅ Found {len(retrieved_docs)} relevant documents")
                        
            top_docs = rerank_docs(llm, refined_query, retrieved_docs)[:3]
            context = "\n\n".join([doc.page_content for doc, _ in top_docs])
            
            final_prompt = f"Answer the query using the context:\nQuery: {user_query}\nContext: {context}"
            answer = llm.invoke(final_prompt).content.strip()
            
            st.markdown("### 📝 Answer:")
            st.write(answer)
        else:
            st.error("❌ No relevant documents found.")
elif submit_button and not user_query:
    st.warning("⚠️ Please enter a query first!")

# ==========================================
# --- Footer ---
# ==========================================
st.markdown("---")