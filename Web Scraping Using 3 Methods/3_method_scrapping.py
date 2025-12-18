# --- Importing Lib's ---
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, UnstructuredURLLoader, SeleniumURLLoader

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title='Web Scraper', layout='wide')

# --- CONFIGURATION & CONSTANTS ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1'
]

# --- SIDEBAR UI ---
st.sidebar.title('Scraper Options')
scraper_choice = st.sidebar.selectbox('Choose Scraper', ('Unstructured-URL Loader', 'Selinum-URL Loader', 'WebBase Loader'))
user_agents = st.sidebar.selectbox('Choose Agent: ', USER_AGENTS)

# --- MAIN App UI ---
st.title('🌐 LangChain Web Scraper')
url = st.text_area('Enter Website Url: ')

# --- SCRAPING LOGIC ---
if st.button('Scrape!'):
    with st.spinner('Scraping...'):
        if not url:
            st.error('No url found! try again.')
        else:
            try:
                docs = [] 
                # Option 1: WebBaseLoader (uses BeautifulSoup)
                if scraper_choice == 'WebBase Loader':
                    loader = WebBaseLoader(url, header_template={'User-Agent': user_agents})
                    docs = loader.load()
                
                # Option 2: SeleniumURLLoader (for JavaScript-heavy websites)
                elif scraper_choice == 'Selinum-URL Loader':
                    loader = SeleniumURLLoader(urls=[url]) 
                    docs = loader.load()
                
                # Option 3: UnstructuredURLLoader (good for various document types/layouts)
                elif scraper_choice == 'Unstructured-URL Loader':
                    loader = UnstructuredURLLoader(urls=[url], headers={'User-Agent': user_agents})
                    docs = loader.load() 

                if docs:
                    content = '\n'.join([doc.page_content for doc in docs])
                    
                    st.subheader('Code Preview:-')
                    st.write(content[:1000] + ("..." if len(content) > 1000 else ""))

                    with st.expander('Expand to See Full Content'):
                        st.text_area('Scraped Text', content, height=300)
                        st.download_button('Download here', content, 'scraped_text.txt')
                else:
                    st.warning('No content extracted. The site might be blocking bots or have no text.')
            
            except Exception as e:
                st.error(f'Error Occurred: {e}')