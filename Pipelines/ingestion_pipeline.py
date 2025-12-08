import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ----------------------------- 
# Load Documents
# ----------------------------- 
def load_docs(docs_path='docs'):
    print(f'Loading all docs from {docs_path}')
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f'The directory {docs_path} does not exist')
    
    loader = DirectoryLoader(
        path=docs_path,
        glob='*.txt',
        loader_cls=TextLoader,
        show_progress=True
    )
    docs = loader.load()
    
    if len(docs) == 0:
        raise FileNotFoundError(f'No document files found in {docs_path}')
    
    # Preview first 2 docs
    for i, doc in enumerate(docs[:2]):
        print(f'\nDocument {i + 1}')
        print(f"Source: {doc.metadata['source']}")
        print(f'Content: {doc.page_content[:100]}')
        print(f'MetaData: {doc.metadata}')
    
    return docs

# ----------------------------- 
# Split Documents
# ----------------------------- 
def split_docs(docs, chunk_size=1000, chunk_overlap=0):
    print('\nSplitting documents into chunks')
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    
    # Preview first 4 chunks
    if chunks:
        for i, chunk in enumerate(chunks[:4]):
            print(f'\nChunk {i + 1}')
            print(f"Source: {chunk.metadata['source']}")
            print(f'Length: {len(chunk.page_content)} characters')
            print(f'Content: {chunk.page_content[:100]}')
        
        if len(chunks) > 4:
            print(f'\n... {len(chunks) - 4} more chunks')
    
    print(f'\nTotal chunks created: {len(chunks)}')
    return chunks

# ----------------------------- 
# Create Vector Store
# ----------------------------- 
def create_vectorstore(chunks, persist_dir='./chroma_db'):
    print('\nCreating embeddings and vector store...')
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Change to 'cuda' if GPU available
    )
    
    # Create and persist ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, # Create chunks from documents
        embedding=embeddings, # Then embed them.
        persist_directory=persist_dir
    )
    
    print(f'Persisted to: {persist_dir}')
    return vectorstore

# ----------------------------- 
# Load Existing Vector Store
# ----------------------------- 
def load_vectorstore(persist_dir='./chroma_db'):
    print(f'\nLoading existing vector store from {persist_dir}')
    
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f'Vector store not found at {persist_dir}')
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore

# ----------------------------- 
# Query Vector Store
# ----------------------------- 
def query_vectorstore(vectorstore, query, k=3):
    print(f'\nQuerying: "{query}"')
    print(f'Retrieving top {k} results...\n')
    
    # Cosine Similarity search
    results = vectorstore.similarity_search(query, k=k)
    
    for i, doc in enumerate(results):
        print(f'Result {i + 1}:')
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f'Content: {doc.page_content[:200]}...')
    return results

# ----------------------------- 
# Main Function
# ----------------------------- 
def main():
    # Load and process documents
    docs = load_docs(docs_path='docs')
    chunks = split_docs(docs, chunk_size=1000, chunk_overlap=100)
    
    # Create vector store
    vectorstore = create_vectorstore(chunks, persist_dir='./chroma_db')
    
    # Example query
    print('TESTING RETRIEVAL')
    query_vectorstore(vectorstore, "What is the main topic?", k=3)
    

if __name__ == '__main__':
    main()