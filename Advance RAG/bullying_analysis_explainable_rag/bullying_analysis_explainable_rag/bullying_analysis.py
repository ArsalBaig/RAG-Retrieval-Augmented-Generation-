# ==========================================
# CORE CONFIGURATION & IMPORTS
# ==========================================

import os
import pandas as pd
import streamlit as st
import zipfile

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# ==========================================
# --- Path Configuration ---
# ==========================================
dataset_path = 'Advance RAG\\explainable_rag_dataset\\Bullying_2018.csv.zip'

# ==========================================
# --- Data Extraction and Data Transformation ---
# ==========================================

@st.cache_resource
def extract_csv_from_zip(zip_path, output_directory= 'extracted_data'):
    if not os.path.exists(zip_path):
        st.error(f'The file not found! {os.path.abspath(zip_path)}')
        st.stop()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(output_directory)
    csv_file_name = 'Bullying_2018.csv'
    csv_file_path = os.path.join(output_directory, csv_file_name)

    df = pd.read_csv(csv_file_path, sep=';')
    df = df.fillna('Not Reported')
    
    documents = []
    for _, row in df.iterrows():
        content = f"""Student Record:
        - Age: {row.get('Custom_Age', 'Not Reported')}
        - Sex: {row.get('Sex', 'Not Reported')}
        - Close Friends: {row.get('Close_friends', 'Not Reported')}
        - Bullied on School Property: {row.get('Bullied_on_school_property_in_past_12_months', 'Not Reported')}
        - Cyber Bullied: {row.get('Cyber_bullied_in_past_12_months', 'Not Reported')}
        - Felt Lonely: {row.get('Felt_lonely', 'Not Reported')}
        - Physically Attacked: {row.get('Physically_attacked', 'Not Reported')}
        - Physical Fighting: {row.get('Physical_fighting', 'Not Reported')}
        - Missed Classes: {row.get('Missed_classes_or_school_without_permission', 'Not Reported')}
    """
        metadata = {
            'record_id': str(row.get('record', 'Unknown')),
            'age': str(row.get('Custom_Age', 'Not Reported')),
            'sex': str(row.get('Sex', 'Not Reported')),
            'close_friends': str(row.get('Close_friends', 'Not Reported')),
            'bullied_school': str(row.get('Bullied_on_school_property_in_past_12_months', 'Not Reported')),
            'cyber_bullied': str(row.get('Cyber_bullied_in_past_12_months', 'Not Reported')),
            'felt_lonely': str(row.get('Felt_lonely', 'Not Reported'))
        }
        documents.append(Document(page_content=content, metadata=metadata))    
    return documents

# ==========================================
# --- Vector Store Construction & Embedding ---
# ==========================================

@st.cache_resource
def create_vectorstore_and_embedding(_documents): 
    persist_dir = 'chroma_bullying_analysis'
    embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 1. Check if directory exists and has files to avoid re-indexing
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
        st.info(f"Loaded existing vectorstore.")
    else:
        # 2. Only index if database doesn't exist. 
        st.info(f"Indexing {len(_documents)} documents. This may take a few minutes...")
        vectorstore = Chroma.from_documents(
            documents=_documents[:2000], # Limiting to 2000 for performance
            embedding=embedding,
            persist_directory=persist_dir
        )
        st.success('Indexing completed!')
    return vectorstore, embedding

# ==========================================
# --- Initilizing LLM ---
# ==========================================

@st.cache_resource
def load_llm():
    llm = ChatGroq(
        model= 'llama-3.3-70b-versatile',
        temperature= 0,
        api_key= os.getenv('GROQ_API_KEY')
    )
    return llm

# ==========================================
# --- Fetching Relevant Docs ---
# ==========================================

def get_relevant_documents(query, vectorstore):
# MMR=  Maximal Marginal Relevance, which balances relevance and diversity in retrieved results.
# fetch_k= Number of top documents to fetch before applying MMR re-ranking.
    retriever = vectorstore.as_retriever(search_type= 'mmr', search_kwargs= {'k': 6, 'fetch_k' : 20})
    docs = retriever.invoke(query)
    return docs

# ==========================================
# --- Explainable RAG ---
# ==========================================

def explainable_rag(query, llm, vectorstore):
    docs = get_relevant_documents(query, vectorstore)
    context = []
    evidence = []

    for idx, doc in enumerate(docs, 1): # idx will start from 1 for human readable format.
        context.append(f'Record {idx}: {doc.page_content}')
        evidence.append({
            'record_num': idx,
            'age': doc.metadata.get('age', 'Unknown'),
            'sex': doc.metadata.get('sex', 'Unknown'),
            'close_friends': doc.metadata.get('close_friends', 'Unknown'),
            'bullied_school': doc.metadata.get('bullied_school', 'Unknown'),
            'cyber_bullied': doc.metadata.get('cyber_bullied', 'Unknown'),
            'felt_lonely': doc.metadata.get('felt_lonely', 'Unknown')
        })
    context = "\n\n".join(context)
    prompt = f'''You are an expert Data Analyst. Answer based ONLY on the provided evidence.
    Instructions:
    - Use the evidence to answer the question directly.
    - Do NOT make assumptions or use outside knowledge.
    - If you cannot find the answer, say "The data does not contain this information."
    - Be concise and specific, referencing the evidence clearly.
    
    Question: {query}
    Evidence from Dataset: {context}
    Detailed Answer:'''
    response = llm.invoke(prompt)
    return response.content.strip(), evidence

# ==========================================
# --- Streamlit UI Setup ---
# ==========================================

st.set_page_config(page_icon='🛡️', page_title='Bullying Analysis Explainable RAG', layout= 'centered')
st.title('🛡️ Bullying Analysis in School (Explainable RAG)')

user_query = st.text_input('Enter your question about bullying analysis?', placeholder='Example: What factors are associated with students feeling lonely?etc')
if st.button('Get Answer'):
    if user_query:
        with st.spinner('Processing your request...'):
            documents = extract_csv_from_zip(dataset_path)
            vectorstore, embedding = create_vectorstore_and_embedding(documents)
            llm = load_llm()
            answer, evidence = explainable_rag(user_query, llm, vectorstore)

            st.subheader('Answer:')
            st.write(answer)
            st.subheader('Evidence:')
            for idx, ev in enumerate(evidence, 1):
                st.write(f'Evidence {idx}: {ev}')
    else:
        st.error('⚠️ Please enter a question to proceed.')
else:
    st.info('Enter a question and click "Get Answer" to see the analysis.')

# ==========================================
# --- Footer ---
# ==========================================

st.markdown("---")
st.markdown("Developed by Arsal Baig")