import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import tempfile, os
from pathlib import Path
from typing import List, Dict


@st.cache_resource
def initialize_llm():
    """Initialize the LLM and chat interface"""
    llm = HuggingFaceEndpoint(
        # repo_id="meta-llama/Llama-3.1-8B-Instruct", # Old----
        # repo_id="meta-llama/Llama-3.1-405B-Instruct", # Old----
        repo_id="meta-llama/Llama-3.1-70B-Instruct",
        task="conversational",
        temperature=0.7,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


@st.cache_resource
def get_embeddings():
    """Initialize embeddings model"""
    return SentenceTransformerEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"trust_remote_code": True},
    )
    
    # # lower version of embedding -----------------------------------
    # return HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    # )


def process_pdf(uploaded_file, use_advanced_loader: bool = False) -> tuple:
    """
    Process uploaded PDF and create vector store
    Returns: (vectorstore, document_count, error_message)
    """
    try:
        # temp storage for pdfs, but you can also use `lazy_load()` while load doc
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        
        # TODO
        
        # # Load PDF with appropriate loader
        # if use_advanced_loader:
        #     # UnstructuredPDFLoader can handle tables and images better
        #     loader = UnstructuredPDFLoader(tmp_path, mode="elements")
        # else:
        #     loader = PyPDFLoader(tmp_path)
        
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        doc_splits = text_splitter.split_documents(docs)
        
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(doc_splits, embeddings)
        
        # temp storage for pdfs, you can also use `lazy_load()`
        os.unlink(tmp_path)
        
        return vectorstore, len(doc_splits), None
        
    except Exception as e:
        return None, 0, str(e)


def get_answer_with_sources(question: str, vectorstore, llm) -> Dict:
    """
    Get answer with source citations
    Returns: dict with 'answer', 'sources', and 'source_documents'
    """
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}
        )
        
        relevant_docs = retriever.invoke(question)
        
        # Create context from documents
        context = "\n\n".join([
            f"[Source {i+1}]:\n{doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful educational assistant. Your role is to:
1. Explain concepts in simple, easy-to-understand language
2. Break down complex topics into digestible parts
3. Use examples and analogies when helpful
4. Keep explanations concise but comprehensive
5. Always mention which source ([Source 1], [Source 2], etc.) supports your statements

Be friendly, encouraging, and make learning enjoyable!"""),
            ("user", """Question: {question}

Context from the document:
{context}

Please provide a clear, simple explanation based on the context above. Mention the source numbers when you reference information.""")
        ])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "context": context
        })
        
        # Prepare sources
        sources = []
        for i, doc in enumerate(relevant_docs):
            source_info = {
                'number': i + 1,
                'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                'page': doc.metadata.get('page', 'N/A')
            }
            sources.append(source_info)
        
        return {
            'answer': answer,
            'sources': sources,
            'source_documents': relevant_docs
        }
        
    except Exception as e:
        return {
            'answer': f"Sorry, I encountered an error: {str(e)}",
            'sources': [],
            'source_documents': []
        }

def display_sources(sources: List[Dict]):
    """Display source citations in an organized way"""
    if sources:
        st.markdown("### 📖 Sources from Document")
        for source in sources:
            with st.expander(f"📄 Source {source['number']} (Page {source['page']})"):
                st.markdown(f"```\n{source['content']}\n```")