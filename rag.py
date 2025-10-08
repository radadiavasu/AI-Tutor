import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
import os
import tempfile
from pathlib import Path
from typing import List, Dict
import time
from sentence_transformers import SentenceTransformer
from comp.indexes import *

# Page configuration
st.set_page_config(
    page_title="📚 AI Learning Assistant",
    # page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        background-color: #121110;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e8eaf6;
        border-left: 4px solid #3f51b5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-msg {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# all session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
    




# App starts---------------------------------------------------------------------------------

st.title("📚 AI Learning Assistant")
st.markdown("*Learn = Earn,  Learn smarter, not harder*")

with st.sidebar:
    st.header("📁 Upload Your Study Material")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload the PDF you want to learn from"
    )
    
    # TODO
    
    # use_advanced = st.checkbox(
    #     "Advanced PDF Processing",
    #     help="Better handling of tables and images (slower)",
    #     value=False
    # )
    
    if uploaded_file:
        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("📖 Reading and understanding your document..."):
                vectorstore, chunk_count, error = process_pdf(uploaded_file) #, use_advanced
                
                if error:
                    st.error(f"❌ Error processing PDF: {error}")
                else:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    
                    st.markdown(f"""
                    <div class="success-msg">
                        ✅ Successfully processed!<br>
                        📄 <b>{uploaded_file.name}</b><br>
                        📊 {chunk_count} chunks created
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()
    
    if st.session_state.pdf_processed:
        st.success(f"✅ Ready: {st.session_state.current_pdf_name}")
        
        if st.button("🗑️ Clear & Upload New PDF"):
            st.session_state.vectorstore = None
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.current_pdf_name = None
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Learning")
    st.markdown("""
    - Ask specific questions about topics you don't understand
    - Request examples or analogies
    - Ask to simplify complex explanations
    - Request summaries of lengthy sections
    """)

# Main chat interface
if st.session_state.pdf_processed and st.session_state.vectorstore:
    # st.markdown("### 💬 You share you learn.")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            if chat.get('sources'):
                with st.expander("📚 View Sources"):
                    display_sources(chat['sources'])
    
    # Chat input
    user_question = st.chat_input("Type your question here...")
    
    if user_question:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                llm = initialize_llm()
                result = get_answer_with_sources(
                    user_question,
                    st.session_state.vectorstore,
                    llm
                )
                
                st.write(result['answer'])
                
                if result['sources']:
                    with st.expander("📚 View Sources"):
                        display_sources(result['sources'])
        
        # Save to chat history
        st.session_state.chat_history.append({
            'question': user_question,
            'answer': result['answer'],
            'sources': result['sources']
        })
        
        st.rerun()
        
    if st.session_state.chat_history:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    

else:
    st.markdown("""
    ### Example Quesions for quick start...
    - "Explain [complex topic] in simple terms"
    - "What does [term] mean?"
    - "Give me examples of [concept]"
    - "Summarize the section about [topic]"
    - "What are the key points in this document?"
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with ❤️ for better learning | Powered by LangChain & HuggingFace
</div>
""", unsafe_allow_html=True)