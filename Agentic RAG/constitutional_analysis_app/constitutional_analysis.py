# ==========================================
# Importing Lib's
# ==========================================

import os
import cv2
import pytesseract
import streamlit as st
from typing import List
from dotenv import load_dotenv
# LangChain & LangGraph components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document 

# ==========================================
# Path Configuration
# ==========================================

load_dotenv()
pdf_path = r'Agentic RAG\constitutional_data'
os.makedirs(pdf_path, exist_ok=True)

# ==========================================
# Function to process both PDF text and Image OCR
# ==========================================

def extract_text_from_pdf_and_images() -> List[Document]:
    all_docs = []
    if not os.path.exists(pdf_path):
        return all_docs
    
    image_extensions = ('.jpg', '.jpeg', '.png')
    for file in os.listdir(pdf_path):
        file_lower = file.lower()
        path = os.path.join(pdf_path, file)
        
        if file_lower.endswith('.pdf'):
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
        
        elif file_lower.endswith(image_extensions):
            img = cv2.imread(path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) 
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) # Doing morphological operations   
            gray = cv2.dilate(gray, kernel, iterations=1)
            gray = cv2.erode(gray, kernel, iterations=1)
            
            text = pytesseract.image_to_string(gray)
            if text.strip():
                all_docs.append(Document(page_content=text, metadata={"source": file}))
    return all_docs

# ==========================================
# Ingestion Pipeline
# ==========================================

def ingestion_pipeline():
    raw_documents = extract_text_from_pdf_and_images()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory='./chroma_db'
    )
    vectorstore.persist()
    print('✅ Ingestion complete. Vector store saved.')    

# ==========================================
# Cached Resource Initializations
# ==========================================

@st.cache_resource
def load_vectordb_and_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

llm = load_llm()
loaded_vectorstore = load_vectordb_and_embeddings()

# ==========================================
# Setup Retrieval QA Chain
# ==========================================

retriever = loaded_vectorstore.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
# ==========================================
# STREAMLIT UI DESIGN 
# ==========================================
st.set_page_config(page_title="Constitutional Agentic RAG", layout="wide")
st.title("⚖️ Constitutional Agentic RAG Application")

st.sidebar.header('📤 Upload Legal Files')
uploaded_files = st.sidebar.file_uploader(
    'Upload PDF or Image files:',
    type=['pdf', 'jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

# ==========================================
# File Uploading Section.
# ==========================================

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(pdf_path, file.name), 'wb') as f:
            f.write(file.getbuffer())
    ingestion_pipeline()
    load_vectordb_and_embeddings.clear() # Clear cache so the new files are recognized by the vectorstore.
    st.success('✅ Files uploaded and ingested successfully!')


user_question = st.text_input('Ask a question about the legal documents of Pakistan')
if st.button('🚀 Analyze!') and user_question:
    with st.spinner('Analyzing...'):
        result = qa_chain({"query": user_question})
        st.markdown("### 📝 Answer:")
        st.write(result["result"])

        st.markdown("### 📚 Source Documents:")
        for doc in result["source_documents"]:
            st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')} | "
                     f"**Page:** {doc.metadata.get('page', 'N/A')}")
            st.write(doc.page_content[:400])
            st.markdown("---")
# ==========================================
# Footer 
# ==========================================

st.markdown("---")