# ==========================================
# CORE CONFIGURATION & IMPORTS
# ==========================================

import os
import zipfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

load_dotenv()

DATA_ZIP_PATH = "Advance RAG/cross_encoder_dataset/corpus_train.csv.zip"
DATA_CSV_PATH = "corpus_train.csv"
CHROMA_PERSIST_DIR = 'cross_encoder_chroma_db'

# ---------------------------------------
# LOAD CSV DATASET (OPTIMIZED)
# ---------------------------------------

@st.cache_resource
def load_my_dataset():
    if not os.path.exists(DATA_CSV_PATH):
        with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")

    df = pd.read_csv(DATA_CSV_PATH, dtype=str)
# Define the columns to check for titles and text content.
    documents = []
    title_cols = ["title", "claim"]
    text_cols = ["text", "abstract", "content", "body", "evidence"]

    # Find which columns exist (do this once)
    title_col = None
    text_col = None

# Loops through the 'title_cols' and then searches col in dataframe, if find it then store it and breaks.    
    for col in title_cols:
        if col in df.columns:
            title_col = col
            break
    
    for col in text_cols:
        if col in df.columns:
            text_col = col
            break

    if title_col and text_col:
        df['combined'] = df[title_col].fillna('') + '\n\n' + df[text_col].fillna('')
    elif text_col:
        df['combined'] = df[text_col].fillna('')
    elif title_col:
        df['combined'] = df[title_col].fillna('')
    
    documents = df['combined'].str.strip().tolist()

    filtered_docs = []
    for doc in documents:
# Checks if doc exists and isn't just empty spaces.        
        if doc and doc.strip(): 
            filtered_docs.append(doc)
    documents = filtered_docs

    if not documents:
        raise ValueError("No usable text columns found.")   
    return documents

# ---------------------------------------
# ------ CHUNKING  --------
# ---------------------------------------

@st.cache_resource
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    all_chunks = []
    batch_size = 1000
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size] # [0: 0 + 1000] = first 1000 docs.
        for doc in batch:
            all_chunks.extend(splitter.split_text(doc))
    return all_chunks
# ---------------------------------------
# --------- VECTORSTORE -----------
# ---------------------------------------

@st.cache_resource
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'batch_size': 128,  
            'normalize_embeddings': True 
        }
    )
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        batch_size = 500  
        vectorstore = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
# Checks if it's the first batch. If not then add it to the exsisting vectorstore.            
            if vectorstore is None:
                vectorstore = Chroma.from_texts(
                    batch,
                    embeddings,
                    persist_directory=CHROMA_PERSIST_DIR
                )
            else:
                vectorstore.add_texts(batch)    
    return vectorstore

# ---------------------------------------
# -------- CROSS ENCODER -----------
# ---------------------------------------

# Cross Encoder is used to score the relevance b/w a query and a document.
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------
# --------- RETRIEVAL -----------
# ---------------------------------------

def retrieve(query, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 30}
    )
    return retriever.invoke(query)

# ---------------------------------------
# ------------ RE-RANKING -----------
# ---------------------------------------

def rerank(query, docs, encoder):
    if not docs:
        return []
# Query-document pairs.    
    pairs = [(query, doc.page_content) for doc in docs]
    scores = encoder.predict(pairs, batch_size=32) 

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:5]

# ---------------------------------------
# ---------- LLM -----------
# ---------------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

# ---------------------------------------
# ----------- CONTEXT BUILDER -----------
# ---------------------------------------

def build_context(ranked_docs):
    return "\n\n".join(doc.page_content for doc, _ in ranked_docs)

# ---------------------------------------
# ----------- RAG PIPELINE -----------
# ---------------------------------------

def run_rag(query, vectorstore, reranker, llm):
    retrieved_docs = retrieve(query, vectorstore)
    reranked_docs = rerank(query, retrieved_docs, reranker)

    prompt = f"""
Answer the question using ONLY the context below.
-Instructions:
- If the answer can be found in the context, provide a concise and accurate response.
- If the answer cannot be found in the context, say "I don't know".
- Do not assume any information that is not explicitly stated in the context.
- If you found the answer, then explain it within 5-6 lines.

Context:
{build_context(reranked_docs)}

Question:
{query}
"""
    response = llm.invoke(prompt)
    return response.content.strip(), retrieved_docs, reranked_docs

# ---------------------------------------
# ----------- STREAMLIT UI -----------
# ---------------------------------------
st.set_page_config(
    page_title="Cross-Encoder Re-Ranking RAG",
    page_icon="📚",
    layout="centered"
)
st.title("📚 Cross-Encoder Re-Ranking RAG")

with st.sidebar:
    st.header('Demo Examples')
    examples = [
        'Can regular exercise improve cognitive function in older adults?',
        'Do face masks reduce the transmission of respiratory viruses?',
        'Is coffee consumption associated with increased heart disease risk?'
    ]
# Explicitly enumerating from idx 1.
    for idx, example in enumerate(examples, 1):
        st.markdown(f'Example {idx} : {example}')
        
with st.spinner("Loading dataset & models..."):
    documents = load_my_dataset()
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)
    reranker = load_reranker()
    llm = load_llm()

query = st.text_input(
    "Enter your question:",
    placeholder="Can regular exercise improve mental health?"
)

if st.button('Get Answer'):
    if query:
        with st.spinner("Running RAG pipeline..."):
            answer, retrieved_docs, reranked_docs = run_rag(
                query, vectorstore, reranker, llm
            )
        st.subheader("🤖 Final Answer")
        st.success(answer)

st.markdown("---")
st.markdown("Developed by **Arsal Baig**")