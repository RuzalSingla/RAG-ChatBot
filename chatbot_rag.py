import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("Chatbot")

# initialize pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "sampleindex"
index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# initialize llm
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))

# initialize chat history (no system message yet — added dynamically per query)
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history (skip SystemMessages — they're internal)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# chat input
prompt = st.chat_input("Ask me anything!")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(prompt))

    # retrieve relevant docs
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # build system prompt with fresh context
    system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}"""

    system_prompt_fmt = system_prompt.format(context=docs_text)

    # place system message at the front, don't append it to history
    messages_to_send = [SystemMessage(system_prompt_fmt)] + st.session_state.messages

    # get response
    result = llm.invoke(messages_to_send).content

    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state.messages.append(AIMessage(result))