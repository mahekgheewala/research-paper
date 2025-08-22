from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import requests
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict, Tuple, Optional
from PyPDF2 import PdfReader
import logging
from datetime import datetime
import time
import json
from urllib.parse import urljoin, urlparse, quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import hashlib
from threading import Lock
from dotenv import load_dotenv
from flask_cors import CORS
import uuid

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Google Gemini imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# -------------------------------------------------
# Load env & create app
# -------------------------------------------------
load_dotenv()
app = Flask(__name__)

from flask_cors import CORS

CORS(app, supports_credentials=True, origins=[
    "http://localhost:3000",  # React frontend (local)
])

# Secret key for Flask cookie-based sessions
app.secret_key = os.environ.get('SECRET_KEY', 'PuDhFAut9DtJz7_9X2tVABtND40INHBKDLtNNcjhAE0')

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None",  # Needed for cross-site requests
    SESSION_COOKIE_SECURE=True,      # Required with SameSite=None (since Render is HTTPS)
    MAX_CONTENT_LENGTH=16 * 1024 * 1024
)


# Allowed frontend origins - Updated for local development
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.46:3000",
    "https://research-paper-frontend-1.onrender.com",  # replace with your actual frontend Render URL
    # "https://research-paper-2.onrender.com"  # deployed frontend (Render)
]

# CORS - Fixed configuration
CORS(
    app,
    resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,
    methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization']
)

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
class Config:
    """Configuration settings for the arXiv Q&A system"""
    PDF_DOWNLOAD_DIR = "arxiv_papers"  # Changed to relative path
    FAISS_INDEX_DIR = "faiss_index"    # Changed to relative path
    METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "document_metadata.json")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 6
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_MODEL = "gemini-1.5-flash"
    MAX_PAPERS = 8
    ARXIV_BASE_URL = "http://export.arxiv.org/api"
    SEARCH_DELAY = 2
    MAX_RETRIES = 3
    GEMINI_API_KEY = "AIzaSyCkR49rt17EAidFlimh4RS8HOOVVbtUsAM"

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def get_api_key():
    if not Config.GEMINI_API_KEY:
        logger.warning("⚠️ Gemini API key not found in environment variable GEMINI_API_KEY")
        raise ValueError("No API key provided. Please set GEMINI_API_KEY environment variable.")
    return Config.GEMINI_API_KEY

def create_directories():
    os.makedirs(Config.PDF_DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(Config.FAISS_INDEX_DIR, exist_ok=True)
    logger.info(f"Created directories: {Config.PDF_DOWNLOAD_DIR}, {Config.FAISS_INDEX_DIR}")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text.strip()

# -------------------------------------------------
# arXiv Paper Fetching
# -------------------------------------------------
def search_arxiv_papers(query: str, max_results: int = Config.MAX_PAPERS) -> List[Dict]:
    logger.info(f"Searching arXiv for: '{query}'")
    papers_metadata = []
    try:
        search_url = f"{Config.ARXIV_BASE_URL}/query?search_query={quote(query)}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        logger.info(f"Searching URL: {search_url}")
        response = requests.get(search_url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            for i, entry in enumerate(entries):
                arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                paper_info = {
                    'arxiv_id': arxiv_id,
                    'title': entry.find('atom:title', ns).text if entry.find('atom:title', ns) else 'Unknown Title',
                    'summary': entry.find('atom:summary', ns).text if entry.find('atom:summary', ns) else '',
                    'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                    'published': entry.find('atom:published', ns).text[:4] if entry.find('atom:published', ns) else '',
                    'pdf_url': pdf_url,
                    'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
                    'local_path': None,
                    'citations': '0'
                }
                papers_metadata.append(paper_info)
                logger.info(f"Extracted paper {i+1}: {paper_info.get('title', 'Unknown title')[:50]}...")
        else:
            logger.error(f"Error searching arXiv: {response.status_code} - {response.text}")
        logger.info(f"Successfully extracted {len(papers_metadata)} papers")
    except Exception as e:
        logger.error(f"Error searching arXiv: {str(e)}")
    return papers_metadata

def download_selected_papers(papers_metadata: List[Dict], selected_indices: List[int]) -> List[Dict]:
    selected_papers = []
    logger.info(f"Fetching {len(selected_indices)} selected papers")
    for idx in selected_indices:
        if idx >= len(papers_metadata):
            logger.warning(f"Index {idx} is out of range for papers list")
            continue
            
        paper = papers_metadata[idx]
        logger.info(f"Fetching: {paper['title']}")
        try:
            full_content = ""
            local_path = None
            if paper['pdf_url']:
                response = requests.get(paper['pdf_url'], timeout=30)
                if response.status_code == 200:
                    local_path = os.path.join(Config.PDF_DOWNLOAD_DIR, f"{paper['arxiv_id']}.pdf")
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    with pdfplumber.open(local_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                full_content += page_text + "\n"
            if not full_content:
                full_content = paper.get('summary', 'Content not available')
            paper['full_content'] = clean_text(full_content)
            paper['local_path'] = local_path if local_path else 'limited_content'
            selected_papers.append(paper)
            logger.info(f"Successfully fetched content: {paper['title'][:50]}...")
        except Exception as e:
            logger.error(f"Failed to fetch content for {paper['arxiv_id']}: {str(e)}")
            continue
    logger.info(f"Successfully processed {len(selected_papers)} papers")
    return selected_papers

# -------------------------------------------------
# Content Processing
# -------------------------------------------------
def load_and_process_data(papers_metadata: List[Dict]) -> List[Dict]:
    logger.info("Processing paper content and creating text chunks")
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    for paper in papers_metadata:
        if not paper.get('full_content') and not paper.get('local_path'):
            logger.warning(f"No content available for paper: {paper['title']}")
            continue
        logger.info(f"Processing: {paper['title']}")
        full_text = paper.get('full_content', paper.get('summary', ''))
        if not full_text.strip():
            logger.warning(f"No text content for: {paper['title']}")
            continue
        enhanced_text = f"""
Title: {paper['title']}
Authors: {', '.join(paper.get('authors', []))}
Published: {paper.get('published', 'Unknown')}
arXiv ID: {paper.get('arxiv_id', 'Unknown')}

Content:
{full_text}
        """.strip()
        chunks = text_splitter.split_text(enhanced_text)
        for i, chunk in enumerate(chunks):
            content_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
            chunk_data = {
                'content': chunk,
                'metadata': {
                    'title': paper['title'],
                    'arxiv_id': paper.get('arxiv_id', 'unknown'),
                    'authors': paper.get('authors', []),
                    'published': paper.get('published', 'Unknown'),
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'source': f"{paper['title']} (arXiv ID: {paper.get('arxiv_id', 'unknown')})",
                    'source_type': 'arxiv',
                    'arxiv_url': paper.get('arxiv_url', ''),
                    'doc_id': f"{paper.get('arxiv_id', 'unknown')}_{content_hash}"
                }
            }
            all_chunks.append(chunk_data)
        logger.info(f"Created {len(chunks)} chunks for {paper['title']}")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

# -------------------------------------------------
# Embeddings and Vector Store
# -------------------------------------------------
def load_document_metadata() -> Dict:
    metadata_path = Config.METADATA_FILE
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load document metadata: {str(e)}")
            return {}
    return {}

def save_document_metadata(metadata: Dict) -> None:
    metadata_path = Config.METADATA_FILE
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved document metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save document metadata: {str(e)}")

def create_embeddings(chunks: List[Dict]) -> FAISS:
    logger.info("Creating or updating embeddings and building vector store")
    if not chunks:
        raise ValueError("No chunks provided for embedding creation")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = load_existing_vectorstore()
    existing_metadata = load_document_metadata()
    existing_doc_ids = set(existing_metadata.keys())
    new_chunks = []
    new_metadata = {}
    for chunk in chunks:
        doc_id = chunk['metadata']['doc_id']
        if doc_id not in existing_doc_ids:
            new_chunks.append(chunk)
            new_metadata[doc_id] = chunk['metadata']
    if new_chunks:
        logger.info(f"Found {len(new_chunks)} new chunks to embed")
        new_texts = [chunk['content'] for chunk in new_chunks]
        new_metadatas = [chunk['metadata'] for chunk in new_chunks]
        if vectorstore is None:
            logger.info(f"Generating embeddings for {len(new_texts)} new chunks")
            vectorstore = FAISS.from_texts(
                texts=new_texts,
                embedding=embeddings,
                metadatas=new_metadatas
            )
        else:
            logger.info(f"Adding {len(new_texts)} new embeddings to existing vector store")
            vectorstore.add_texts(
                texts=new_texts,
                metadatas=new_metadatas
            )
        existing_metadata.update(new_metadata)
        save_document_metadata(existing_metadata)
        vectorstore.save_local(Config.FAISS_INDEX_DIR)
        logger.info(f"Vector store updated and saved to {Config.FAISS_INDEX_DIR}")
    else:
        logger.info("No new chunks to embed; using existing vector store")
    if vectorstore is None:
        raise ValueError("Failed to create or load vector store")
    return vectorstore

def load_existing_vectorstore() -> Optional[FAISS]:
    try:
        if os.path.exists(os.path.join(Config.FAISS_INDEX_DIR, "index.faiss")):
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            vectorstore = FAISS.load_local(Config.FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing vector store")
            return vectorstore
    except Exception as e:
        logger.error(f"Failed to load existing vector store: {str(e)}")
    return None

# -------------------------------------------------
# Gemini LLM Integration
# -------------------------------------------------
class GeminiLLM:
    def __init__(self, api_key: str, model_name: str = Config.GEMINI_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        logger.info(f"Initialized Gemini LLM: {model_name}")

    def __call__(self, inputs):
        if isinstance(inputs, dict):
            if 'context' in inputs and 'question' in inputs:
                prompt = f"Context: {inputs['context']}\n\nQuestion: {inputs['question']}\n\nAnswer:"
            else:
                prompt = inputs.get('prompt', inputs.get('text', str(inputs)))
        else:
            prompt = str(inputs)
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "I apologize, but I couldn't generate a response for your question."
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            return f"Error generating response: {str(e)}"

def initialize_gemini_llm():
    logger.info(f"Initializing Gemini LLM: {Config.GEMINI_MODEL}")
    try:
        gemini_llm = GeminiLLM(get_api_key(), Config.GEMINI_MODEL)
        logger.info("Gemini LLM initialized successfully")
        return gemini_llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
        raise

# -------------------------------------------------
# Enhanced Q&A Chain
# -------------------------------------------------
class EnhancedConversationalQA:
    def __init__(self, retriever, gemini_llm):
        self.retriever = retriever
        self.gemini_llm = gemini_llm
        self.conversation_history = []

    def __call__(self, inputs):
        original_question = inputs.get('query', inputs.get('question', ''))
        docs = self.retriever.get_relevant_documents(original_question)
        
        # Create context from documents
        context_parts = []
        for doc in docs:
            if hasattr(doc, 'metadata'):
                title = doc.metadata.get('title', 'Unknown Paper')
                context_parts.append(f"Paper: {title}\nContent: {doc.page_content}")
        
        context = '\n\n'.join(context_parts)
        
        enhanced_prompt = f"""
You are an expert research assistant analyzing academic papers from arXiv. Answer the user's question using ONLY the provided research paper content.

RESEARCH PAPERS CONTEXT:
{context}

USER QUESTION: {original_question}

INSTRUCTIONS:
1. Answer based ONLY on the provided research papers
2. Be comprehensive and detailed in your analysis
3. Include specific citations in format: [Paper Title]
4. If the papers don't contain enough information, state this clearly

ANSWER:
"""
        try:
            answer = self.gemini_llm.generate(enhanced_prompt)
            unique_sources = []
            for doc in docs:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source_info = doc.metadata['source']
                    if source_info not in unique_sources:
                        unique_sources.append(source_info)
            if unique_sources:
                answer += f"\n\n📚 Sources:\n" + "\n".join(f"• {source}" for source in unique_sources[:5])
            return {'result': answer, 'source_documents': docs}
        except Exception as e:
            logger.error(f"Error in enhanced QA: {str(e)}")
            return {'result': f"Error processing question: {str(e)}", 'source_documents': docs}

def build_enhanced_chain(vectorstore: FAISS) -> EnhancedConversationalQA:
    logger.info("Building Enhanced QA chain with Gemini LLM")
    gemini_llm = initialize_gemini_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
    qa_chain = EnhancedConversationalQA(retriever, gemini_llm)
    logger.info("Enhanced QA chain with Gemini built successfully")
    return qa_chain

def ask_question(chain, user_query: str) -> str:
    logger.info(f"Processing question: {user_query}")
    try:
        result = chain({'query': user_query})
        if isinstance(result, dict):
            answer = result.get('result', result.get('answer', 'No answer generated'))
        else:
            answer = str(result)
        return answer
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return f"Error processing your question: {str(e)}"

# -------------------------------------------------
# Session Manager
# -------------------------------------------------
class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.lock = Lock()
        self.vectorstore = None

    def create_session(self, session_id: str):
        with self.lock:
            if session_id not in self.sessions:
                logger.info(f"🆕 Creating new session: {session_id}")
                self.sessions[session_id] = {
                    'status': 'new',
                    'created_at': datetime.now().isoformat(),
                    'papers': [],
                    'selected_papers': [],
                    'conversation_history': [],
                    'qa_chain': None,
                    'chunks_count': 0
                }

    def get_session(self, session_id: str):
        with self.lock:
            return self.sessions.get(session_id)

    def update_session(self, session_id: str, data: dict):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(data)

    def delete_session(self, session_id: str):
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"🗑️ Deleted session {session_id}")

    def get_vectorstore(self):
        with self.lock:
            if self.vectorstore is None:
                self.vectorstore = load_existingにつながるexisting_vectorstore()
            return self.vectorstore

    def update_vectorstore(self, chunks: List[Dict]):
        with self.lock:
            self.vectorstore = create_embeddings(chunks)
            return self.vectorstore

session_manager = SessionManager()

# -------------------------------------------------
# Flask session management
# -------------------------------------------------
@app.before_request
def ensure_session():
    if request.method == "OPTIONS":
        return ("", 200)
    sid = session.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
        session["session_id"] = sid
        logger.info(f"🆕 Assigned session cookie {sid}")
    if not session_manager.get_session(sid):
        session_manager.create_session(sid)

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "session_id": session.get("session_id"),
        "active_sessions": len(session_manager.sessions)
    })

@app.route('/api/get_session_status', methods=['GET'])
def get_session_status():
    sid = session.get("session_id")
    sdata = session_manager.get_session(sid)
    if not sdata:
        return jsonify({
            "success": True,
            "status": "no_session",
            "papers_count": 0,
            "selected_papers_count": 0,
            "qa_ready": False,
            "chunks_count": 0
        })
    return jsonify({
        "success": True,
        "status": sdata.get("status", "unknown"),
        "papers_count": len(sdata.get("papers", [])),
        "selected_papers_count": len(sdata.get("selected_papers", [])),
        "qa_ready": sdata.get("qa_chain") is not None and sdata.get("status") == "ready",
        "chunks_count": sdata.get("chunks_count", 0)
    })

@app.route('/api/search_papers', methods=['POST'])
def search_papers_route():
    try:
        sid = session.get("session_id")
        logger.info(f"🔍 Search papers request - Session ID: {sid}")
        payload = request.get_json(force=True) or {}
        logger.info(f"📥 Search request payload: {payload}")
        query = (payload.get('query') or '').strip()
        max_results = int(payload.get('max_results') or Config.MAX_PAPERS)
        if not query:
            logger.warning(f"⚠️ Empty search query received for session {sid}")
            return jsonify({'success': False, 'error': 'Please provide a search query'})
        
        logger.info(f"🔍 [{sid}] Searching arXiv for: {query}")
        papers = search_arxiv_papers(query, max_results)
        logger.info(f"📚 Found {len(papers)} papers for query: {query}")
        
        if not papers:
            logger.warning(f"⚠️ No papers found for query: {query}")
            return jsonify({'success': False, 'error': 'No papers found. Try different keywords.'})
        
        # Store papers in session manager
        session_manager.update_session(sid, {
            'papers': papers,
            'search_query': query,
            'status': 'papers_found'
        })
        
        # Also store in Flask session as backup
        session['papers'] = papers
        session['search_query'] = query
        session['status'] = 'papers_found'
        
        logger.info(f"✅ Session after storage - papers count: {len(papers)}")
        
        formatted = []
        for i, paper in enumerate(papers):
            authors = paper.get('authors', [])
            authors_str = ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else '')
            summary = paper.get('summary', '') or ''
            formatted.append({
                'id': i,
                'title': paper.get('title', 'Unknown Title'),
                'authors': authors_str,
                'published': paper.get('published', 'Unknown'),
                'arxiv_id': paper.get('arxiv_id', ''),
                'summary': (summary[:300] + '...') if len(summary) > 300 else summary,
                'arxiv_url': paper.get('arxiv_url', ''),
                'pdf_url': paper.get('pdf_url', '')
            })
        
        return jsonify({'success': True, 'papers': formatted, 'total': len(formatted)})
    except Exception as e:
        logger.exception(f"❌ Error in /search_papers for session {sid}")
        return jsonify({'success': False, 'error': f'Search error: {str(e)}'})

@app.route('/api/select_papers', methods=['POST'])
def select_papers():
    try:
        sid = session.get('session_id')
        sdata = session_manager.get_session(sid)
        
        # Try to get papers from session manager first, then Flask session
        all_papers = sdata.get('papers') if sdata else session.get('papers')
        
        if not all_papers:
            logger.error(f"❌ No papers found in session for SID: {sid}")
            return jsonify({'success': False, 'error': 'No papers found in session. Please try searching again.'})
        
        payload = request.get_json(force=True) or {}
        selected_indices = payload.get('selected_papers') or []
        
        if not selected_indices:
            return jsonify({'success': False, 'error': 'No papers selected'})
        
        logger.info(f"Processing {len(selected_indices)} selected papers")
        
        # Download and process selected papers
        selected_papers = download_selected_papers(all_papers, selected_indices)
        
        if not selected_papers:
            logger.error(f"❌ No papers were successfully processed for session {sid}")
            return jsonify({'success': False, 'error': 'No papers were successfully processed'})
        
        chunks = load_and_process_data(selected_papers)
        
        if not chunks:
            logger.error(f"❌ No text chunks created for session {sid}")
            return jsonify({'success': False, 'error': 'No text chunks were created from the papers'})
        
        vectorstore = session_manager.update_vectorstore(chunks)
        qa_chain = build_enhanced_chain(vectorstore)
        
        session_manager.update_session(sid, {
            'selected_papers': selected_papers,
            'status': 'ready',
            'qa_chain': qa_chain,
            'chunks_count': len(chunks)
        })
        
        formatted_papers = [{
            'title': paper.get('title', 'Unknown Title'),
            'arxiv_id': paper.get('arxiv_id', ''),
            'authors': ', '.join(paper.get('authors', [])[:2]) + ('...' if len(paper.get('authors', [])) > 2 else '')
        } for paper in selected_papers]
        
        return jsonify({
            'success': True,
            'papers_info': formatted_papers,
            'chunks_count': len(chunks)
        })
    except Exception as e:
        logger.exception(f"❌ Error in /select_papers for session {sid}")
        return jsonify({'success': False, 'error': f'Paper selection failed: {str(e)}'})

@app.route('/api/ask_question', methods=['POST'])
def ask_question_endpoint():
    try:
        payload = request.get_json(force=True) or {}
        question = (payload.get('question') or '').strip()
        
        if not question:
            return jsonify({'success': False, 'error': 'Please provide a question'})
        
        sid = session.get('session_id')
        sdata = session_manager.get_session(sid)
        
        if not sdata or not sdata.get('qa_chain'):
            return jsonify({'success': False, 'error': 'Please search and select papers first'})
        
        if sdata.get('status') != 'ready':
            return jsonify({'success': False, 'error': 'System not ready. Please process papers first.'})
        
        start = time.time()
        answer = ask_question(sdata['qa_chain'], question)
        took = time.time() - start
        
        hist = sdata.get('conversation_history', [])
        hist.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'processing_time': took
        })
        
        session_manager.update_session(sid, {
            'conversation_history': hist
        })
        
        return jsonify({'success': True, 'answer': answer, 'processing_time': round(took, 2)})
    except Exception as e:
        logger.exception(f"❌ Error in /ask_question for session {sid}")
        return jsonify({'success': False, 'error': f'Question processing error: {str(e)}'})

@app.route('/api/get_conversation_history', methods=['GET'])
def get_conversation_history():
    sid = session.get('session_id')
    sdata = session_manager.get_session(sid)
    if not sdata:
        return jsonify({'success': False, 'error': 'Session not found'})
    history = sdata.get('conversation_history', [])
    return jsonify({'success': True, 'conversation_history': history[-10:]})

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    old_sid = session.get('session_id')
    if old_sid:
        session_manager.delete_session(old_sid)
    session.clear()
    new_sid = str(uuid.uuid4())
    session['session_id'] = new_sid
    session_manager.create_session(new_sid)
    return jsonify({'success': True, 'message': 'Session cleared and restarted'})

# -------------------------------------------------
# Error handlers
# -------------------------------------------------
@app.errorhandler(404)
def not_found_error(_):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(err):
    logger.exception("Internal server error")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# -------------------------------------------------
# Initialize and run
# -------------------------------------------------
def initialize_app():
    try:
        create_directories()
        get_api_key()
        logger.info("✅ Application initialized")
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to initialize app: {str(e)}")
        return False

if __name__ == '__main__':
    if initialize_app():
        print("🚀 Starting arXiv + Gemini Q&A Flask Application on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    else:
        print("❌ Failed to initialize application")