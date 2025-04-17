Harsha's Chatbot
🚀 100% FREE, Private (No Internet) Harsha’s Advanced RAG Chatbot: Boost Your Chatbot with Hybrid Retrieval (BM25 + FAISS) + Neural Reranking + HyDE!
Harsha's Chatbot enables fast, accurate, and explainable retrieval of information from PDFs, DOCX, and TXT files using DeepSeek-7B, BM25, FAISS, Neural Reranking (Cross-Encoder), GraphRAG, and Chat History Integration. It runs entirely offline, ensuring your data remains private.
🔥 Features

Hybrid Retrieval: Combines BM25 (keyword search) + FAISS (semantic search) for better accuracy.
GraphRAG Integration: Builds a Knowledge Graph from your documents for contextual and relational understanding.
Neural Reranking: Uses Cross-Encoder (ms-marco-MiniLM-L-6-v2) to rank retrieved documents by relevance.
Query Expansion (HyDE): Expands queries using Hypothetical Document Embeddings for better recall.
Chat Memory: Maintains context with chat history for coherent responses.
Document Source Tracking: Displays the source PDF/DOCX/TXT file for answers.
Offline Operation: No internet required, ensuring privacy.

🛠️ Installation
Prerequisites

Hardware: At least 8GB RAM (16GB recommended), CUDA-enabled GPU for acceleration (optional).
Software: Python 3.8+, Git, Docker (optional), Ollama.

Steps

Clone the Repository:
git clone https://github.com/YOUR_GITHUB_USERNAME/Harshas-Chatbot.git
cd Harshas-Chatbot


Set Up Virtual Environment:
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate


Install Dependencies:
pip install --upgrade pip
pip install -r requirements.txt


Install Ollama:

Download and install Ollama from https://ollama.com/.
Pull the required models:ollama pull deepseek-r1:7b
ollama pull nomic-embed-text




Run the Chatbot:
streamlit run app.py


Open your browser at http://localhost:8501 to access the chatbot UI.



Docker Installation

Build and Run:docker-compose up --build


Access the chatbot at http://localhost:8501.

🚀 Usage

Upload Documents: Add PDFs, DOCX, or TXT files via the sidebar.
Ask Questions: Enter queries in the chat interface.
View Responses: The chatbot retrieves relevant document chunks, processes them with GraphRAG, and generates answers using DeepSeek-7B.

🤝 Contributing
Fork this repo, submit pull requests, or open issues for new features or bug fixes. Feedback is welcome on Reddit!
📜 Credits
Developed by Harsha. © All Rights Reserved 2025.
