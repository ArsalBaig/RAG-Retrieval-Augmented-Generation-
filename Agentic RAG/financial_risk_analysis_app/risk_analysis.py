# ==========================================
# 1. CORE CONFIGURATION & IMPORTS
# ==========================================
import os
import cv2
import pytesseract
import streamlit as st
from PIL import Image
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph components
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Path Configuration ---

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
load_dotenv()

# ==========================================
# 2. CACHED RESOURCE INITIALIZATION
# ==========================================

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_vectorstore(_embed):
    """Connects to the local Chroma vector database."""
    return Chroma(
        persist_directory="./receipt_chroma_db",
        embedding_function=_embed
    )

# Load resources into memory
llm = load_llm()
embeddings = load_embeddings()
vectorstore = get_vectorstore(embeddings)

# ==========================================
# 3. OCR & DATA INGESTION LOGIC
# ==========================================

RECEIPT_DIR = 'Agentic RAG\Images_data' 

def extract_text_from_receipts() -> List[str]:

    texts = []
    if not os.path.exists(RECEIPT_DIR):
        return texts

    for file in os.listdir(RECEIPT_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(RECEIPT_DIR, file)
            img = cv2.imread(path)
            # 1.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Rescale Image (Makes small text larger for Tesseract)
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            
            # 3. Apply Dilation/Erosion to remove noise cleans up background spots.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            gray = cv2.dilate(gray, kernel, iterations=1)
            gray = cv2.erode(gray, kernel, iterations=1)
            
            # 4.
            text = pytesseract.image_to_string(gray)
            texts.append(text)
    return texts

def ingest_receipts():

    texts = extract_text_from_receipts()
    # To avoid short or empty texts.
    docs = [Document(page_content=t) for t in texts if len(t.strip()) > 20]
    if docs:
        vectorstore.add_documents(docs)
        vectorstore.persist() # This saves the indexed data to hard-disk.

# Initialize data once per session
if "ingested" not in st.session_state:
    ingest_receipts()
    st.session_state.ingested = True

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ==========================================
# 4. AGENT SYSTEM & PROMPTS
# ==========================================

# -- Agent State Definition --
class AgentState(TypedDict):
    query: str      
    plan: str       
    context: str    
    analysis: str   
    verified: str   

# -- Prompt Templates --

planner_prompt = PromptTemplate(
    input_variables=["query"],
    template="Plan steps to analyze spending risk for: {query}. Return a concise list."
)

analysis_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""Analyze the following receipts for financial risk:
    Context: {context}
    Query: {query}
    Focus on: Overspending, Patterns, and Unusual Anomalies."""
)

verification_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="Review this financial analysis for logical errors: {analysis}"
)

# -- Agent Functions (Nodes) --
def planner_agent(state: AgentState):
    res = llm.invoke(planner_prompt.format(query=state["query"]))
    return {"plan": res.content}

def retrieval_agent(state: AgentState):
    docs = retriever.invoke(state["query"]) 
    context = "\n\n".join(d.page_content for d in docs) if docs else "No data."
    return {"context": context}

def analysis_agent(state: AgentState):
    res = llm.invoke(analysis_prompt.format(query=state["query"], context=state["context"]))
    return {"analysis": res.content}

def verification_agent(state: AgentState):
    res = llm.invoke(verification_prompt.format(analysis=state["analysis"]))
    return {"verified": res.content}

# ==========================================
# 5. WORKFLOW GRAPH CONSTRUCTION
# ==========================================

workflow = StateGraph(AgentState)

# Add agents as nodes
workflow.add_node("planner", planner_agent)
workflow.add_node("retriever", retrieval_agent)
workflow.add_node("analyzer", analysis_agent)
workflow.add_node("verifier", verification_agent)

# Define sequence: Planner -> Retriever -> Analyzer -> Verifier
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "analyzer")
workflow.add_edge("analyzer", "verifier")

app = workflow.compile()

# ==========================================
# 6. STREAMLIT UI (PRESENTATION LAYER)
# ==========================================

# --- Page Setup ---
st.set_page_config(page_title="Risk Analyst", page_icon="💸", layout="wide")
st.title("💸 Agentic Financial Risk Analyst")

# --- File Upload Section ---
uploaded_files = st.file_uploader("Upload receipt images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs(RECEIPT_DIR, exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join(RECEIPT_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())
    ingest_receipts()
    st.success("Receipts indexed Successfully!")

# --- Query Section ---
st.subheader("🔎 Step 2: Ask Your Question")
query = st.text_input("What would you like to know?", placeholder="Analyze my food spending risk...")

if st.button("🚀 Analyze Risk") and query:
    with st.spinner("Agentic team is processing your data..."):
        result = app.invoke({"query": query}) # Sending user query to the app workflow.

    st.markdown("---")
    st.subheader("📊 Financial Risk Analysis Result")
    
    if "analysis" in result:
        st.write(result["analysis"])
    else:
        st.warning("The analysis could not be completed. Check if receipts were uploaded.")
    st.success("Task Complete.")

# --- Footer ---
st.markdown("---")