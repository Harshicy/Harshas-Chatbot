import streamlit as st
import requests
import json
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from sentence_transformers import CrossEncoder
import torch
import os
from dotenv import load_dotenv, find_dotenv
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Load environment variables first
load_dotenv(find_dotenv())

# Streamlit configuration (must be the first Streamlit command)
st.set_page_config(page_title="Harsha's Chatbot", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)

# Fix for torch classes not found error
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Configure API URLs with fallbacks
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:7b")
EMBEDDINGS_MODEL = "nomic-embed-text:latest"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Cross-Encoder with offline support
reranker = None
try:
    # Load model without use_auth_token (use cached files if available)
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    st.success("CrossEncoder model loaded successfully.")
except Exception as e:
    st.warning(f"Failed to load CrossEncoder model: {str(e)}. Running without reranking. Pre-cache the model locally with `python -c \"from sentence_transformers import CrossEncoder; CrossEncoder('{CROSS_ENCODER_MODEL}')\"` or check cache at https://huggingface.co/docs/transformers/installation#offline-mode.")
    reranker = None

# Manage Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            try:
                process_documents(uploaded_files, reranker, EMBEDDINGS_MODEL, OLLAMA_API_URL)
                st.success("Documents processed successfully!")
                st.session_state.documents_loaded = True
                st.session_state.retrieval_pipeline = retrieve_documents  # Assume this is set after processing
            except requests.exceptions.ConnectionError as e:
                st.error(f"Connection error during document processing: {str(e)}. Ensure OLLAMA_API_URL is reachable.")
            except Exception as e:
                st.error(f"Document processing failed: {str(e)}")

    st.markdown("---")
    st.header("⚙️ RAG Settings")
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True if reranker else False)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Footer Credits
    st.sidebar.markdown("""
        <div style="position: absolute; bottom: 10px; right: 10px; font-size: 12px; color: gray;">
            <b>Developed by:</b> Harsha © All Rights Reserved 2025
        </div>
    """, unsafe_allow_html=True)

# Chat Interface
st.title("🤖 Harsha's Chatbot")
st.caption("Advanced RAG Chatbot with GraphRAG, Hybrid Retrieval, Neural Reranking, and Chat History")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about your documents..."):
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])  # Last 5 messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Build context with RAG
        context = ""
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                docs = st.session_state.retrieval_pipeline(prompt, OLLAMA_API_URL, MODEL, chat_history)
                context = "\n".join(
                    f"[Source {i+1}]: {doc.page_content}" 
                    for i, doc in enumerate(docs[:st.session_state.max_contexts])
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
                context = "No context available due to retrieval failure."
        
        # Structured prompt
        system_prompt = f"""Use the chat history to maintain context:
            Chat History:
            {chat_history}

            Analyze the question and context through these steps:
            1. Identify key entities and relationships
            2. Check for contradictions between sources
            3. Synthesize information from multiple contexts
            4. Formulate a structured response

            Context:
            {context}

            Question: {prompt}
            Answer:"""
        
        # Setup retry session for API call
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Make API request with retry
        try:
            response = session.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL,
                    "prompt": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": st.session_state.temperature,
                        "num_ctx": 4096
                    }
                },
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    token = data.get("response", "")
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                    if data.get("done", False):
                        break
            response_placeholder.markdown(full_response)
            
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection error to Ollama API: {str(e)}. Ensure OLLAMA_API_URL is set to a reachable server.")
            full_response = "Sorry, I couldn’t connect to the AI server. Please check your configuration."
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            full_response = "An error occurred while processing your request."
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            full_response = "An unexpected error occurred."
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})