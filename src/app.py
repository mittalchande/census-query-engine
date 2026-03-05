import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API Keys
load_dotenv()

# Page Configuration
st.set_page_config(page_title="CensusInsight AI", layout="centered")
st.title("Census Data Explorer")
st.markdown("Query official census documents using Natural Language.")

# 1. Model & Embedding Setup
# Using Llama-3.1-8b on Groq for sub-second inference speeds
llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'), 
    model_name="llama-3.1-8b-instant",
    temperature=0.1 # Low temperature for factual census data
)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

# 2. Vector Store Management (Cached for Speed)
@st.cache_resource 
def get_vectorstore():
    persist_dir = "./chroma_db_census"
    
    # Load existing database if it exists to skip re-indexing
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        embeddings = get_embeddings()
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # First-time indexing logic
    if not os.path.exists("./data"):
        st.error("Please create a '/data' folder and add your Census PDFs.")
        return None

    with st.spinner("Analyzing Census Documents... This happens only once."):
        loader = PyPDFDirectoryLoader("./data") 
        docs = loader.load()
     
        # Chunking optimized for text-heavy reports
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        return Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )

# Initialize or Load Vector Store
if "vector_db" not in st.session_state:
    st.session_state.vector_db = get_vectorstore()

# 3. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. RAG Chain Construction
retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

template = """
You are a Census Data Analyst. Use the following context to answer the question concisely.
If the answer is not in the context, say you don't know. 
Keep your answer to 2-3 sentences maximum.

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Modern LCEL Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. The User Interface
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
user_query = st.chat_input("Ask about population, demographics, or trends...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate and Stream Assistant Response
    with st.chat_message("assistant"):
        response = st.write_stream(rag_chain.stream(user_query))
        
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Source Transparency
    with st.sidebar:
        st.header("Sources Found")
        source_docs = retriever.invoke(user_query)
        for i, doc in enumerate(source_docs):
            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page_num = doc.metadata.get('page', 'N/A')
            st.write(f"**{i+1}. {source_name} (Page {page_num})**")
            st.caption(doc.page_content[:150] + "...")