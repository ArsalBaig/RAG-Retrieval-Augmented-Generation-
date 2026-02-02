# ==========================================
# CORE CONFIGURATION & IMPORTS
# ==========================================
import os
import pandas as pd
import streamlit as st
import re

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
# ==========================================
# --- Data Loading and Transformation ---
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(r'Advance RAG\time_series_dataset\QS World University Rankings 2025 (Top global universities).csv', encoding='latin1')
    df.columns = df.columns.str.strip() # removes extra spaces in column names.
    df = df.dropna(subset=['Institution_Name', 'RANK_2025', 'RANK_2024'])
    
    def parse_rank(rank_str):
        if pd.isna(rank_str):
            return 9999
    # Check if rank is a text/string. And contains '-' or '='.
        if isinstance(rank_str, str):
            if '-' in rank_str:
                return int(rank_str.split('-')[0]) # If '-' take first part only.
            rank_str = rank_str.replace('=', '').strip() # If '=' remove it.
        try:
            return int(rank_str)
        except:
            return 9999
    
    df['Rank_2025_Numeric'] = df['RANK_2025'].apply(parse_rank)
    df['Rank_2024_Numeric'] = df['RANK_2024'].apply(parse_rank)
    
    documents = []
    
    # For 2025 documents
    for _, row in df.iterrows():
        content = (
            f"In 2025, {row['Institution_Name']} ranked {row['RANK_2025']} globally. "
            f"Location: {row['Location']}, Region: {row['Region']}. "
            f"In 2025, it was ranked {row['RANK_2025']}."
        )
        
        metadata = {
            'Year': 2025,
            'Institution_Name': row['Institution_Name'],
            'Rank': row['RANK_2025'],
            'Rank_Numeric': row['Rank_2025_Numeric'],
            'Country': row['Location'],
            'Region': row['Region']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # 2024 documents
    for _, row in df.iterrows():
            content = (
                f"In 2024, {row['Institution_Name']} ranked {row['RANK_2024']} globally. "
                f"Location: {row['Location']}, Region: {row['Region']}. "
                f"In 2024, it was ranked {row['RANK_2024']}."
            )
            
            metadata = {
                'Year': 2024,
                'Institution_Name': row['Institution_Name'],
                'Rank': row['RANK_2024'],
                'Rank_Numeric': row['Rank_2024_Numeric'],
                'Country': row['Location'],
                'Region': row['Region']
            }
            documents.append(Document(page_content=content, metadata=metadata))    
    return df, documents
df, documents = load_data()

# ==========================================
# --- Embedding & Vector Store Setup ---
# ==========================================
@st.cache_resource
def get_embedding_and_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        collection_name="university_rankings_db"
    )
    return embeddings, vectorstore

# ==========================================
# --- LLM Initilization ---
# ==========================================
@st.cache_resource
def load_llm():
    llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.0,
        api_key=os.getenv('GROQ_API_KEY')
    )
    return llm

# ==========================================
# --- Reranking & Query Handling ---
# ==========================================
def extract_score(response_text):
# Matches numbers 1-10.
    numbers = re.findall(r'\b([1-9]|10)\b', response_text) #\b is word boundary
    if numbers:
        return int(numbers[0])
    return 5

def rerank_docs(llm, query, docs):
    ranked_docs = []
    for doc in docs:
        prompt = f'''Based on the query "{query}", rate the relevance of this document on a scale of 1 to 10.
        Document: {doc.page_content}
        Respond with only a number from 1 to 10, where:
        10 = Highly relevant
        5 = Moderately relevant  
        1 = Not relevant
        '''
        
        try:
            response = llm.invoke(prompt)
            res_score = extract_score(response.content.strip())
        except Exception as e:
            print(f"Error during reranking: {e}. Using default score.")
            res_score = 5
# Assign relevance score to document metadata.
        doc.metadata['relevance_score'] = res_score
        ranked_docs.append(doc)
    ranked_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
    return ranked_docs

# ==========================================
# --- Top N Universities Extraction ---
# ==========================================

def get_top_n_universities(df, year, n=10):
    if year == 2025:
# Gets n rows with smallest Rank_2025_Numeric values.
        top_unis = df.nsmallest(n, 'Rank_2025_Numeric')
        return top_unis[['Institution_Name', 'RANK_2025', 'RANK_2024', 'Location', 'Region']]
    else:  # 2024
        top_unis = df.nsmallest(n, 'Rank_2024_Numeric')
        return top_unis[['Institution_Name', 'RANK_2024', 'RANK_2025', 'Location', 'Region']]

# ==========================================
# --- Temporal RAG Function(Time-Base-RAG)---
# ==========================================

def temporal_rag(query, year):
    embeddings, vectorstore = get_embedding_and_vectorstore()
    llm = load_llm()
    
    # Looking for sepecific keywords in query.
    is_top_query = any(word in query.lower() for word in ['top', 'best', 'highest', 'leading'])
    
    if is_top_query:
        numbers = re.findall(r'\b(\d+)\b', query)
        n = int(numbers[0]) if numbers else 10
        n = min(n, 50)  # Cap at 50
        
        top_unis = get_top_n_universities(df, year, n)
        
        if year == 2025:
            context = "\n".join([
                f"{row['Institution_Name']} is ranked {row['RANK_2025']} in 2025 (was {row['RANK_2024']} in 2024). Location: {row['Location']}, Region: {row['Region']}"
                for _, row in top_unis.iterrows()
            ])
        else:
            context = "\n".join([
                f"{row['Institution_Name']} is ranked {row['RANK_2024']} in 2024 (became {row['RANK_2025']} in 2025). Location: {row['Location']}, Region: {row['Region']}"
                for _, row in top_unis.iterrows()
            ])
# LLM Prompt.        
        prompt = f'''You are a university ranking analysis expert. Based on the provided ranking data, answer the user's query in a detailed, informative paragraph.

Ranking Data for {year}:
{context}

User Query: {query}

Instructions:
- Include specific university names, their exact rankings, and locations
- Mention how rankings changed from the previous year when relevant
- Write in a clear, informative style
- List the universities in order of their ranking
- Make the response engaging and educational
'''
    else:
        temporal_query = f'{query} in {year}'
        
        all_docs = vectorstore.similarity_search(temporal_query, k=20)
        year_filtered_docs = [doc for doc in all_docs if doc.metadata.get('Year') == year]
        
        if not year_filtered_docs:
            year_filtered_docs = all_docs[:10] # Takes top 10 from all_docs if none match the year.
        
        if not year_filtered_docs: # If still empty then print below.
            return "No relevant documents found for your query."

        reranked_docs = rerank_docs(llm, temporal_query, year_filtered_docs[:10])
        
        top_docs = reranked_docs[:5]
        context = "\n".join([f"{doc.page_content}" for doc in top_docs])
# LLM Prompt.
        prompt = f'''You are a university ranking analysis expert. Based on the provided context about university rankings, answer the user's query in a detailed, informative paragraph.

Context (University Ranking Information for {year}):
{context}

User Query: {query}

Instructions:
- Include specific university names, their rankings, and locations
- Write in a clear, informative style
- Make the response engaging and educational
'''    
    response = llm.invoke(prompt).content.strip()
    return response

# ==========================================
# --- Streamlit UI ---
# ==========================================
st.set_page_config(page_title="Global University Ranking Analysis RAG", layout="wide")
st.title("🌍 Global University Ranking Analysis RAG")

with st.sidebar:
    st.header('⚙️ Settings')
    selected_year = st.selectbox('Select Year', options=[2025, 2024])
    st.markdown('---')
    
    st.subheader('📊 Dataset Info')
    st.write(f"Total Universities: {len(df)}")
    st.write(f"Countries: {df['Location'].nunique()}")
    st.write(f"Regions: {df['Region'].nunique()}")
    
    st.markdown('---')
    st.subheader('💡 Example Queries')
    st.markdown("""
    - Top 10 universities in the world
    - Best universities in the United States
    - Top universities in Asia
    - Universities ranked between 50-100
    - Which universities improved their ranking?
    """)

user_query = st.text_input('Enter your ranking analysis query:')

if st.button('🔍 Analyze', type='primary'):
    if user_query.strip():
        with st.spinner('Generating analysis...'):
            try:
                answer = temporal_rag(user_query, selected_year)
                st.subheader('📝 Analysis Result:')
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
    else:
        st.warning("⚠️ Please enter a query.")

# ==========================================
# --- Footer ---
# ==========================================
st.markdown('---')