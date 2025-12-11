# Importing Lib's
import os
import streamlit as st
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.prompts import PromptTemplate

# Initilizing LLM Model.
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

# Required Fields for Resume Parsing.
PROMPT_TEMPLATE = """
You are an expert resume parser. Given the resume text, extract the following fields and return a single valid JSON object:

{{
  "Name": "...",
  "Email": "...",
  "Phone": "...",
  "LinkedIn": "...",
  "Skills": [...],
  "Education": [...],
  "Experience": [...],
  "Projects": [...],
  "Certifications": [...],
  "Languages": [...]
}}
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["resume_text"])

chain = prompt | llm

def load_resume(uploaded_file):
    temp_path = f'temp_{uploaded_file.name}' # Creates a temporary file path.
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    filename = uploaded_file.name.lower()

    if filename.endswith('.pdf'):
        loader = PyPDFLoader(temp_path)
    elif filename.endswith('.docx'):
        loader = Docx2txtLoader(temp_path)
    elif filename.endswith('.txt'):
        loader = TextLoader(temp_path)
    else:
        return None
    docs = loader.load()
    
    # Cleaning up temp file
    try:
        os.remove(temp_path)
    except:
        pass
    return docs


# Streamlit App.
def main():
    st.set_page_config(page_title="Resume Parser", layout="centered")
    st.title("Resume Parser with Gemini-2.5-Flash")

    uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file:
        with st.spinner('Loading resume...'):
            docs = load_resume(uploaded_file)
            if not docs:
                st.error("Unsupported file format.")
        
        st.subheader("Extracted Resume Information")
        preview_text = '\n'.join([doc.page_content for doc in docs])
        st.text_area("Resume Text Preview", preview_text, height=200)

        if st.button('Parse Resume'):
            with st.spinner('Analyzing resume...'):
                full_text = ' '.join([doc.page_content for doc in docs])
                response_message = chain.invoke({"resume_text": full_text})
                json_string = response_message.content.strip()
                
                # Remove markdown code blocks if present
                if json_string.startswith('```'):
                # Removes the triple backticks and take content of 1st index.
                    json_string = json_string.split('```')[1] 
                    if json_string.startswith('json'):
                        json_string = json_string[4:] # Remove first 4 chars 'json'
                    json_string = json_string.strip() # Remove leading whitespaces/newlines.

                try:
                    parsed_data = json.loads(json_string)
                    st.success("Resume parsed successfully!")
                    st.json(parsed_data)
                except json.JSONDecodeError as e:
                    st.error(f"Failed to decode JSON: {str(e)}")
                    st.subheader("Raw Response:")
                    st.code(json_string, language='text')

if __name__ == "__main__":
    main()