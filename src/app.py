import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="PolicyGuard: Travel Insurance AI", layout="wide")
st.title("🛡️ PolicyGuard AI Assistant with LangChain")

# 1. Setup Models
llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama-3.1-8b-instant")
embeddings = OpenAIEmbeddings()

@st.cache_resource 
def get_vectorstore():
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    with st.spinner("Indexing policies..."):
        loader = DirectoryLoader("./data", glob="./*.pdf", loader_cls=PDFPlumberLoader)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        chunks = splitter.split_documents(docs)
        return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = get_vectorstore()

# 2. Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Sidebar Configuration
with st.sidebar:
    st.header("Search Context")
    policy_filter = st.selectbox(
        "Who is travelling?",
        ["Both (Compare Plans)", "Visitor to Canada", "Canadian Travelling Abroad"]
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 4. Retrieval Logic with Filtering
search_kwargs = {"k": 6, "fetch_k": 20}
if policy_filter == "Visitor to Canada":
    search_kwargs["filter"] = {"source": {"$contains": "visitor-to-canada"}}
elif policy_filter == "Canadian Travelling Abroad":
    search_kwargs["filter"] = {"source": {"$contains": "single-trip-all-inclusive"}}

retriever = st.session_state.vector_db.as_retriever(
    search_type="mmr", 
    search_kwargs=search_kwargs  # IMPORTANT: This activates the filter
)

# 5. System Instructions
system_instructions = (
    "### ROLE\n"
    "You are a Precision Insurance Auditor for Manulife policies. Use ONLY the context provided.\n\n"
    "### DATA EXTRACTION PROTOCOL\n"
    "1. Identify Policy: 'visitor-to-canada' ($25k-$100k limits) vs 'all-inclusive' ($10M limit).\n"
    "2. Verify Type: 'Repatriation' is for the Insured (Actual Cost). 'Bedside Companion' is for friends ($3k).\n"
    "3. Exclusions: For 'Sports' or 'High-Risk', list bullet points. Stop before other sections like 'Pregnancy'.\n\n"
    "### OUTPUT FORMAT\n"
    "**Plan Name**: [Name]\n"
    "**Benefit Limit**: [Exact Amount]\n"
    "**Source**: [Filename], **Page X**"
)

# 6. Chain Construction
prompt = ChatPromptTemplate.from_template(
    system_instructions + "\n\nContext:\n{context}\n\nQuestion: {input}"
)

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask about medical limits or exclusions...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        response = rag_chain.invoke(user_query)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.expander("View Source Document Chunks"):
        source_docs = retriever.invoke(user_query)
        for doc in source_docs:
            file_name = os.path.basename(doc.metadata.get("source", "Unknown"))
            page_num = int(doc.metadata.get("page", 0)) + 1
            st.markdown(f"**Source:** {file_name} | **Page:** {page_num}")
            st.info(doc.page_content)