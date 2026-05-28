# 🤖 RAG ChatBot
<img width="1068" height="823" alt="image" src="https://github.com/user-attachments/assets/3ee44c96-6917-48fc-9438-9901b4e2630f" />

Deployed Link : https://ragchatbot-ruzal.streamlit.app/


A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Pinecone, and Groq. The chatbot answers questions based on documents stored in a Pinecone vector database, using free and open-source tools.

## 🛠️ Tech Stack

- **LangChain** — framework for building LLM applications
- **Pinecone** — vector database for storing and retrieving document embeddings
- **HuggingFace Embeddings** — free local embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **Groq** — free LLM API (LLaMA 3)
- **Streamlit** — web UI for the chatbot

## 📁 Project Structure

```
LangChain-Pinecone-RAG/
│
├── documents/              # Put your PDF files here
├── chatbot_rag.py          # Streamlit chatbot UI
├── ingestion.py            # Load and index PDF documents
├── retrieval.py            # Test retrieval from Pinecone
├── sample_ingestion.py     # Ingest sample hardcoded documents
├── sample_retrieval.py     # Test retrieval with sample documents
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed to git)
└── .env.example            # Example env file
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/RuzalSingla/RAG-ChatBot.git
cd RAG-ChatBot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

```bash
# Windows
venv\Scripts\Activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
pip install langchain-huggingface sentence-transformers langchain-groq
```

### 5. Create accounts and get API keys

- **Pinecone** (vector database): [pinecone.io](https://www.pinecone.io/)
- **Groq** (free LLM API): [console.groq.com](https://console.groq.com)

### 6. Set up your `.env` file

Create a `.env` file in the root directory:

```
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
PINECONE_INDEX_NAME=sampleindex
```

## 🚀 Running the Project

### Ingest sample documents

```bash
python sample_ingestion.py
```

### Ingest your own PDF documents

Place your PDF files in the `documents/` folder, then run:

```bash
python ingestion.py
```

### Test retrieval

```bash
python retrieval.py
```

### Run the chatbot

```bash
streamlit run chatbot_rag.py
```

## 💡 How It Works

1. **Ingestion** — Documents are loaded, split into chunks, and embedded using HuggingFace embeddings, then stored in Pinecone
2. **Retrieval** — When a user asks a question, the most relevant chunks are retrieved from Pinecone using similarity search
3. **Generation** — The retrieved context is passed to the Groq LLM (LLaMA 3) along with the chat history to generate a response

## 📝 Notes

- HuggingFace embeddings run locally — no API key required
- Groq provides a free API for LLaMA 3
- The Pinecone free tier supports up to 2GB of storage
