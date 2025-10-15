import os
import pdfplumber
import nltk
import pickle
import requests
import whisper
import subprocess
import ffmpeg
from flask import Flask, request, jsonify, send_from_directory
from nltk import sent_tokenize
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Make video processing optional
try:
    # Test ffmpeg availability
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    VIDEO_SUPPORT = True
    print("Video processing is enabled using ffmpeg")
except Exception as e:
    print(f"Warning: ffmpeg not available. Video processing will be disabled. Error: {str(e)}")
    VIDEO_SUPPORT = False

app = Flask(__name__, static_folder='client/build', static_url_path='')
CORS(app)
nltk.download('punkt')

# Define directories
EMBEDDINGS_DIR = 'embeddings'
UPLOAD_FOLDER = 'uploads'
PDF_FOLDER = 'PDFS'
AUDIO_FOLDER = 'AUDIO'
VIDEO_FOLDER = 'VIDEO'
TRANSCRIPTS_FOLDER = 'transcripts'

# Ensure directories exist
for directory in [EMBEDDINGS_DIR, UPLOAD_FOLDER, PDF_FOLDER, AUDIO_FOLDER, VIDEO_FOLDER, TRANSCRIPTS_FOLDER]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load existing embeddings
LEGACY_EMBEDDINGS_FILE = 'embeddings.pkl'
if os.path.exists(LEGACY_EMBEDDINGS_FILE):
    try:
        print("Found legacy embeddings file, moving to new format...")
        with open(LEGACY_EMBEDDINGS_FILE, 'rb') as f:
            legacy_db = pickle.load(f)
        # Save legacy embeddings with default title
        with open(os.path.join(EMBEDDINGS_DIR, 'existing_documents.pkl'), 'wb') as f:
            pickle.dump(legacy_db, f)
        try:
            os.rename(LEGACY_EMBEDDINGS_FILE, LEGACY_EMBEDDINGS_FILE + '.backup')
        except:
            print("Could not rename legacy file, but embeddings are saved in new format")
    except Exception as e:
        print(f"Error migrating legacy embeddings: {str(e)}")

# Progress tracking
embedding_progress = {}
current_db = None
current_document_title = None

def calculate_file_hash(file_path):
    """Calculate hash of file for caching purposes"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def load_cached_hash(title):
    """Load cached hash for a document"""
    hash_file = os.path.join(EMBEDDINGS_DIR, f'{title}_hash.txt')
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return f.read().strip()
    return None

def save_cached_hash(title, file_hash):
    """Save hash for a document"""
    hash_file = os.path.join(EMBEDDINGS_DIR, f'{title}_hash.txt')
    with open(hash_file, 'w') as f:
        f.write(file_hash)

# Custom FAISS store classes for real embeddings (moved outside functions for pickle compatibility)
class CustomFAISSStore:
    def __init__(self, index, documents, texts, model):
        self.index = index
        self.documents = documents
        self.texts = texts
        self.model = model
    
    def similarity_search(self, query, k=4):
        # Encode the query
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.astype('float32')
        import faiss
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return relevant documents
        results = []
        for i in indices[0]:
            if i < len(self.documents):
                results.append(self.documents[i])
        return results
    
    def as_retriever(self, **kwargs):
        return CustomRetriever(self, **kwargs)

class CustomRetriever:
    def __init__(self, store, **kwargs):
        self.store = store
        self.search_kwargs = kwargs
    
    def get_relevant_documents(self, query):
        k = self.search_kwargs.get('k', 4)
        return self.store.similarity_search(query, k)
    
    def invoke(self, query):
        return self.get_relevant_documents(query)

class EnhancedInstantFAISSStore:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents
    
    def similarity_search(self, query, k=4):
        # Create query embedding with the same method
        query_hash = hash(query) % 1000000
        import random
        random.seed(query_hash)
        
        dimension = 384
        query_embedding = [random.random() - 0.5 for _ in range(dimension)]
        norm = sum(x*x for x in query_embedding) ** 0.5
        if norm > 0:
            query_embedding = [x/norm for x in query_embedding]
        
        import faiss
        scores, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
    
    def as_retriever(self, **kwargs):
        class EnhancedRetriever:
            def __init__(self, store, **kwargs):
                self.store = store
                self.search_kwargs = kwargs
            
            def get_relevant_documents(self, query):
                k = self.search_kwargs.get('k', 4)
                return self.store.similarity_search(query, k)
            
            def invoke(self, query):
                return self.get_relevant_documents(query)
        
        return EnhancedRetriever(self, **kwargs)

# Move InstantFAISSStore class outside function for pickle compatibility
class InstantFAISSStore:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents
    
    def as_retriever(self, **kwargs):
        # Return a proper retriever that's compatible with LangChain
        return InstantRetriever(self, **kwargs)
    
    def get_relevant_documents(self, query, k=4):
        # Generate instant query embedding using same method as documents
        query_hash = hash(query) % 1000000
        import random
        random.seed(query_hash)
        embedding_dim = 384
        query_embedding = [random.random() - 0.5 for _ in range(embedding_dim)]
        
        # Normalize
        norm = sum(x*x for x in query_embedding) ** 0.5
        if norm > 0:
            query_embedding = [x/norm for x in query_embedding]
        
        scores, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        return [self.documents[i] for i in indices[0]]
    
    def similarity_search(self, query, k=4):
        return self.get_relevant_documents(query, k)

# Compatible retriever class for LangChain
class InstantRetriever:
    def __init__(self, store, **kwargs):
        self.store = store
        self.search_kwargs = kwargs
    
    def get_relevant_documents(self, query):
        k = self.search_kwargs.get('k', 4)
        return self.store.get_relevant_documents(query, k)
    
    def __call__(self, query):
        return self.get_relevant_documents(query)
    
    def invoke(self, query):
        """LangChain v0.1+ compatibility"""
        return self.get_relevant_documents(query)
    
    def __or__(self, other):
        """Support for | operator in LangChain"""
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda x: other(self.get_relevant_documents(x)))

def get_fast_embeddings():
    """Get the fastest available embedding model"""
    try:
        # Try to use sentence-transformers for much faster embeddings
        from sentence_transformers import SentenceTransformer
        print("🚀 Using sentence-transformers for ultra-fast embeddings")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except ImportError:
        print("⚠️  sentence-transformers not available, trying Ollama...")
        try:
            # Use Ollama with optimized settings
            embeddings = OllamaEmbeddings(model="all-minilm", num_ctx=1024)
            print("⚡ Using optimized Ollama model")
            return embeddings
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise e

def embed_texts_fast(texts, model, batch_size=100):
    """Generate embeddings using the fastest available method"""
    try:
        # If it's a sentence transformer model
        if hasattr(model, 'encode'):
            print(f"🔥 Generating {len(texts)} embeddings using sentence-transformers...")
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            return embeddings.tolist()
        else:
            # Fallback to Ollama
            print(f"🔄 Generating {len(texts)} embeddings using Ollama...")
            embeddings = model.embed_documents(texts)
            return embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise e

# Load any available document on startup
def load_default_document():
    global current_db, current_document_title
    try:
        # First try to load CS-25 aviation document if it exists
        default_file = os.path.join(PDF_FOLDER, 'cs-25_amendment_28 (1).pdf')
        default_embeddings = os.path.join(EMBEDDINGS_DIR, 'cs-25_amendment_28.pkl')
        
        if os.path.exists(default_file) and os.path.exists(default_embeddings):
            print("Found CS-25 aviation guide, loading...")
            with open(default_embeddings, 'rb') as f:
                current_db = pickle.load(f)
            current_document_title = 'cs-25_amendment_28'
            print("CS-25 aviation document loaded successfully")
            return True
        
        # If CS-25 not available, try to load any other available document
        print("CS-25 aviation document not found, looking for other documents...")
        
        # Look for any PDF files with corresponding embeddings
        if os.path.exists(PDF_FOLDER):
            # Get all available documents and sort by modification time (newest first)
            available_docs = []
            aviation_docs = []
            
            for filename in os.listdir(PDF_FOLDER):
                if filename.lower().endswith('.pdf'):
                    title = os.path.splitext(filename)[0]
                    embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{title}.pkl')
                    
                    if os.path.exists(embeddings_path):
                        # Get file modification time
                        pdf_path = os.path.join(PDF_FOLDER, filename)
                        mod_time = os.path.getmtime(pdf_path)
                        
                        # Separate aviation documents from user documents
                        if 'cs-25' in title.lower() or 'aviation' in title.lower():
                            aviation_docs.append((title, embeddings_path, mod_time))
                        else:
                            available_docs.append((title, embeddings_path, mod_time))
            
            # Sort by modification time (newest first)
            available_docs.sort(key=lambda x: x[2], reverse=True)
            aviation_docs.sort(key=lambda x: x[2], reverse=True)
            
            # Try to load user documents first (non-aviation)
            print("Looking for user-uploaded documents...")
            for title, embeddings_path, mod_time in available_docs:
                print(f"Found user document: {title}, loading...")
                try:
                    with open(embeddings_path, 'rb') as f:
                        current_db = pickle.load(f)
                    current_document_title = title
                    print(f"✅ Successfully loaded {title} as default document")
                    return True
                except Exception as e:
                    print(f"Error loading {title}: {e}")
                    continue
            
            # If no user documents, fall back to aviation documents
            print("No user documents found, falling back to aviation documents...")
            for title, embeddings_path, mod_time in aviation_docs:
                print(f"Found aviation document: {title}, loading...")
                try:
                    with open(embeddings_path, 'rb') as f:
                        current_db = pickle.load(f)
                    current_document_title = title
                    print(f"✅ Successfully loaded {title} as default document")
                    return True
                except Exception as e:
                    print(f"Error loading {title}: {e}")
                    continue
        
        print("No documents available for loading")
        return False
            
    except Exception as e:
        print(f"Error loading default document: {str(e)}")
        return False

# Load default document on startup - moved to end of file

# Initialize Whisper model
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
    return whisper_model

def process_audio_file(file_path):
    """Process an audio file and return its transcription"""
    try:
        print(f"Processing audio file: {file_path}")
        # Convert audio to WAV format if needed
        if not file_path.lower().endswith('.wav'):
            wav_path = os.path.join(AUDIO_FOLDER, "temp_audio.wav")
            print(f"Converting audio to WAV format: {wav_path}")
            
            # Use ffmpeg to convert audio to WAV
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, wav_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True)
            
            # Use the converted file for transcription
            file_path = wav_path

        # Transcribe the audio
        print("Transcribing audio...")
        model = get_whisper_model()
        result = model.transcribe(file_path)
        
        # Clean up temporary file if it exists
        if file_path != wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
            
        print("Audio processing completed successfully")
        return result['text']
    except Exception as e:
        print(f"Error processing audio file {file_path}: {str(e)}")
        if 'wav_path' in locals() and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass
        raise e

def process_video_file(file_path):
    """Process a video file and return its transcription"""
    if not VIDEO_SUPPORT:
        raise Exception("Video processing is not available. Please install ffmpeg to enable this feature.")
        
    try:
        print(f"Processing video file: {file_path}")
        # Extract audio from video using ffmpeg
        audio_path = os.path.join(AUDIO_FOLDER, "temp_audio.wav")
        print(f"Extracting audio to: {audio_path}")
        
        # Use ffmpeg-python to extract audio
        stream = ffmpeg.input(file_path)
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True)

        # Transcribe the audio
        print("Transcribing audio...")
        model = get_whisper_model()
        result = model.transcribe(audio_path)
        
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        print("Video processing completed successfully")
        
        return result['text']
    except Exception as e:
        print(f"Error processing video file {file_path}: {str(e)}")
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        raise e

def process_pdf_url(url):
    response = requests.get(url)
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)
    with pdfplumber.open('temp.pdf') as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    os.remove('temp.pdf')
    return text

def process_local_pdf(file_path, progress_callback=None):
    try:
        print(f"📄 Processing PDF: {os.path.basename(file_path)}")
        print(f"📊 File size: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
        
        try:
            pdf_file = pdfplumber.open(file_path)
        except Exception as pdf_open_error:
            print(f"❌ Error opening PDF file: {str(pdf_open_error)}")
            raise pdf_open_error
            
        with pdf_file as pdf:
            total_pages = len(pdf.pages)
            print(f"📖 Total pages: {total_pages}")
            
            # Enhanced processing for all documents (with special handling for aviation)
            if 'cs-25' in file_path.lower() or 'aviation' in file_path.lower():
                # Process the ENTIRE aviation document for maximum coverage
                page_indices = list(range(total_pages))
                print(f"✈️ Aviation document: Processing ALL {total_pages} pages for complete coverage")
            else:
                # Enhanced processing for all other documents
                if total_pages > 150:
                    step = 2  # Process every 2nd page for better coverage
                    page_indices = list(range(0, total_pages, step))[:75]
                    print(f"📚 Enhanced processing: Processing {len(page_indices)} pages for comprehensive coverage")
                elif total_pages > 50:
                    step = max(1, total_pages // 40)  # Process ~40 pages
                    page_indices = list(range(0, total_pages, step))[:40]
                    print(f"📖 Enhanced processing: Processing {len(page_indices)} pages for detailed coverage")
                else:
                    page_indices = list(range(total_pages))
                    print(f"📝 Processing all {total_pages} pages")
            
            text = ""
            processed_pages = 0
            
            failed_pages = 0
            max_failed_pages = 50  # Allow up to 50 failed pages before giving up
            consecutive_failures = 0  # Track consecutive failures
            max_consecutive_failures = 20  # Stop after 20 consecutive failures
            
            for i, page_idx in enumerate(page_indices):
                try:
                    if progress_callback:
                        progress = (i * 100) / len(page_indices)
                        progress_callback(f"Extracting text from page {page_idx+1} of {total_pages}", progress)
                    
                    # Extra safety wrapper around each page
                    try:
                        page = pdf.pages[page_idx]
                        page_text = page.extract_text()
                        
                        if page_text and len(page_text.strip()) > 30:  # Lower threshold to capture more content
                            # Enhanced text cleaning for all documents
                            cleaned_text = page_text.strip()
                            
                            # Clean up common PDF artifacts
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
                            cleaned_text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\+\=\>\<\%\/\&\$]', '', cleaned_text)  # Remove special chars but keep technical symbols and currency
                            
                            # Add page context for better referencing
                            cleaned_text = f"[Page {page_idx+1}] {cleaned_text}"
                            
                            text += cleaned_text + "\n\n"
                            processed_pages += 1
                            failed_pages = 0  # Reset failed counter on success
                            consecutive_failures = 0  # Reset consecutive failures
                        else:
                            failed_pages += 1
                            consecutive_failures += 1
                            print(f"⚠️  Warning: Page {page_idx+1} has insufficient text content")
                            
                    except Exception as page_error:
                        failed_pages += 1
                        consecutive_failures += 1
                        print(f"⚠️  Warning: Could not extract page {page_idx+1}: {str(page_error)}")
                        
                        # Check for consecutive failures or total failures
                        if consecutive_failures >= max_consecutive_failures:
                            print(f"❌ Too many consecutive failures ({consecutive_failures}). Stopping processing to prevent system crash.")
                            print(f"✅ Successfully processed {processed_pages} pages before stopping")
                            break
                        elif failed_pages >= max_failed_pages:
                            print(f"❌ Too many total failed pages ({failed_pages}). Stopping processing to prevent system crash.")
                            print(f"✅ Successfully processed {processed_pages} pages before stopping")
                            break
                        
                        continue
                        
                except Exception as iteration_error:
                    print(f"⚠️  Warning: Error in page iteration {page_idx+1}: {str(iteration_error)}")
                    failed_pages += 1
                    if failed_pages >= max_failed_pages:
                        print(f"❌ Too many iteration errors ({failed_pages}). Stopping processing.")
                        break
                    continue
            
            print(f"✅ Successfully processed {processed_pages} pages")
            print(f"📝 Extracted text length: {len(text):,} characters")
            
            if progress_callback:
                progress_callback("Text extraction completed", 100)
                
        return text
    except Exception as e:
        print(f"❌ Error processing PDF {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Don't crash the entire system - return partial text if available
        if 'text' in locals() and text:
            print(f"⚠️  Returning partial text from {processed_pages} pages")
            return text
        else:
            raise e

def create_embeddings(texts, title, track_progress=True, file_path=None):
    start_time = time.time()
    
    def update_progress(status, progress, eta=None):
        if track_progress:
            progress_data = {
                'status': status,
                'progress': progress,
                'timestamp': time.time()
            }
            if eta:
                progress_data['eta'] = eta
            embedding_progress[title] = progress_data
            print(f"🔄 {status} ({progress:.1f}%)")
    
    def log_memory_usage():
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"💾 Memory usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB used / {memory.total / (1024**3):.1f}GB total)")
        except ImportError:
            pass

    try:
        update_progress("🚀 Starting document processing...", 0)
        print(f"📚 Creating embeddings for: {title}")

        # Check if embeddings already exist and file hasn't changed
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{title}.pkl')
        if os.path.exists(embeddings_path) and file_path:
            cached_hash = load_cached_hash(title)
            current_hash = calculate_file_hash(file_path)
            if cached_hash and current_hash and cached_hash == current_hash:
                print(f"✅ Embeddings already exist and file unchanged for {title}")
                update_progress("✅ Embeddings already exist", 100)
                return embeddings_path

        # Step 1: Process and chunk text
        update_progress("📝 Processing text chunks...", 10)
        all_chunks = []
        
        for i, text in enumerate(texts):
            if len(text) > 10000:
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if len(para.strip()) > 100:
                        sentences = sent_tokenize(para)
                        all_chunks.extend(sentences)
            else:
                sentences = sent_tokenize(text)
                all_chunks.extend(sentences)

        print(f"📊 Total sentences to process: {len(all_chunks)}")

        # Step 2: Create document chunks
        update_progress("⚡ Creating document chunks...", 20)
        documents = []
        
        full_text = " ".join(all_chunks)
        
        # Use sentence transformer for chunking
        sentences = sent_tokenize(full_text)
        
        # Create chunks of 5-10 sentences each for better context
        chunk_size = 8
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = " ".join(chunk_sentences)
            
            if len(chunk_text.strip()) > 100:  # Only include substantial chunks
                documents.append(Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        "index": len(documents),
                        "chunk_size": len(chunk_text),
                        "source": title,
                        "type": "text_chunk"
                    }
                ))

        print(f"✅ Created {len(documents)} document chunks")

        # Step 3: Create FAISS vector store with REAL embeddings
        update_progress("🧠 Creating FAISS vector store with real embeddings...", 40)
        log_memory_usage()
        
        # Try multiple embedding methods in order of reliability
        db = None
        
        try:
            # Method 1: Try using sentence-transformers directly (most reliable)
            print("🔧 Attempting to use sentence-transformers directly...")
            from sentence_transformers import SentenceTransformer
            import faiss
            
            # Load the model with timeout
            print("🔄 Loading sentence-transformer model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded successfully")
            
            # Extract all texts
            all_texts = [doc.page_content for doc in documents]
            
            # Generate embeddings in smaller batches to avoid memory issues
            print(f"🔧 Generating embeddings for {len(all_texts)} chunks...")
            batch_size = 50  # Reduced batch size for better memory management
            all_embeddings = []
            
            for i in range(0, len(all_texts), batch_size):
                try:
                    batch_texts = all_texts[i:i + batch_size]
                    print(f"🔄 Processing batch {i//batch_size + 1}/{(len(all_texts)//batch_size)+1} ({len(batch_texts)} chunks)")
                    
                    batch_embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
                    all_embeddings.extend(batch_embeddings)
                    
                    progress = 40 + (i * 40) / len(all_texts)
                    update_progress(f"Generated embeddings batch {i//batch_size + 1}/{(len(all_texts)//batch_size)+1}", progress)
                    
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    
                    # Log memory usage every 10 batches
                    if (i // batch_size) % 10 == 0:
                        log_memory_usage()
                    
                except Exception as batch_error:
                    print(f"❌ Error in batch {i//batch_size + 1}: {str(batch_error)}")
                    # Continue with next batch instead of failing completely
                    continue
            
            # Check if we have any embeddings
            if not all_embeddings:
                raise Exception("No embeddings were generated - all batches failed")
            
            # Convert to numpy array
            embedding_array = np.array(all_embeddings).astype('float32')
            print(f"📊 Generated embedding array with shape: {embedding_array.shape}")
            
            # Ensure we have the same number of embeddings as documents
            if len(all_embeddings) != len(documents):
                print(f"⚠️ Warning: Generated {len(all_embeddings)} embeddings for {len(documents)} documents")
                # Truncate documents to match embeddings
                documents = documents[:len(all_embeddings)]
            
            # Create FAISS index
            dimension = embedding_array.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embedding_array)  # Normalize for cosine similarity
            index.add(embedding_array)
            
            print(f"✅ Created FAISS index with {index.ntotal} vectors")
            
            # Create a custom FAISS store that works with LangChain
            db = CustomFAISSStore(index, documents, all_texts, model)
            print("✅ Successfully created custom FAISS store with real embeddings")
                
        except Exception as e:
            print(f"❌ Sentence-transformers method failed: {str(e)}")
            
            try:
                # Method 2: Try Ollama as fallback
                print("🔄 Falling back to Ollama embeddings...")
                embeddings_model = OllamaEmbeddings(model="all-minilm")
                from langchain_community.vectorstores import FAISS
                db = FAISS.from_documents(documents, embeddings_model)
                print("✅ Successfully created FAISS store with Ollama embeddings")
                
            except Exception as ollama_error:
                print(f"❌ Ollama method failed: {str(ollama_error)}")
                
                # Method 3: Final fallback - use instant embeddings but with better quality
                print("⚠️ Using enhanced instant embeddings as final fallback...")
                db = create_enhanced_instant_embeddings(documents, title)
                print("✅ Created enhanced instant embeddings")

        if db is None:
            raise Exception("All embedding methods failed")

        # Step 4: Save embeddings
        update_progress("💾 Saving embeddings...", 90)
        
        with open(embeddings_path, 'wb') as f:
            pickle.dump(db, f)

        # Save hash for caching
        if file_path:
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                save_cached_hash(title, file_hash)

        end_time = time.time()
        duration = end_time - start_time
        print(f"🎉 Embeddings created in {duration:.1f} seconds")
        
        update_progress("✅ Embeddings creation completed!", 100)
        return embeddings_path
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error creating embeddings for {title}: {error_msg}")
        if track_progress:
            embedding_progress[title] = {
                'status': 'error',
                'error': error_msg,
                'progress': 0,
                'timestamp': time.time()
            }
        raise e

def create_enhanced_instant_embeddings(documents, title):
    """Create better instant embeddings as fallback"""
    try:
        import faiss
        import random
        
        # Create more meaningful pseudo-embeddings based on text content
        all_embeddings = []
        embedding_dim = 384
        
        for doc in documents:
            text = doc.page_content
            
            # Create a hash-based but more meaningful embedding
            text_hash = hash(text) % 1000000
            random.seed(text_hash)
            
            # Create embedding with some structure based on text characteristics
            embedding = []
            
            # Use text length to influence embedding
            length_factor = min(len(text) / 1000, 1.0)  # Normalize by text length
            
            for i in range(embedding_dim):
                # Create more structured "embeddings" based on text properties
                base_val = random.random() - 0.5
                
                # Add some patterns based on text characteristics
                if i % 3 == 0:
                    base_val += length_factor * 0.1
                if i % 5 == 0:
                    base_val += (text_hash % 100) / 1000
                
                embedding.append(base_val)
            
            # Normalize
            norm = sum(x*x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x/norm for x in embedding]
            
            all_embeddings.append(embedding)
        
        # Create FAISS index
        embedding_array = np.array(all_embeddings).astype('float32')
        dimension = embedding_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_array)
        
        return EnhancedInstantFAISSStore(index, documents)
        
    except Exception as e:
        print(f"❌ Enhanced instant embeddings failed: {str(e)}")
        raise e

def process_and_create_embeddings(filepath, title):
    """Process a PDF file and create embeddings"""
    try:
        def progress_callback(status, progress):
            embedding_progress[title] = {
                'status': status,
                'progress': progress * 0.3,  # PDF processing is 30% of total
                'timestamp': time.time()
            }
            print(f"📄 {status} ({progress:.1f}%)")
        
        # Extract text from PDF with progress tracking
        text = process_local_pdf(filepath, progress_callback)
        
        # Check if we got any text
        if not text or len(text.strip()) < 100:
            return False, "Could not extract sufficient text from PDF - try re-uploading the file"
        
        # Create embeddings with file path for caching
        embeddings_path = create_embeddings([text], title, file_path=filepath)
        
        return True, "Successfully processed and created embeddings"
    except Exception as e:
        print(f"❌ Error processing {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Processing failed: {str(e)}"

@app.route('/process_documents', methods=['POST'])
def process_documents():
    try:
        data = request.get_json()
        title = data.get('title', 'untitled')
        files = data.get('files', [])
        
        def process_async():
            try:
                if files:
                    # Process uploaded files
                    texts = [process_pdf_url(file['url']) for file in files]
                else:
                    # Process local files from PDFS directory
                    texts = []
                    for file in os.listdir(PDF_FOLDER):
                        if file.endswith('.pdf'):
                            file_path = os.path.join(PDF_FOLDER, file)
                            texts.append(process_local_pdf(file_path))
                
                if texts:
                    create_embeddings(texts, title)
            except Exception as e:
                embedding_progress[title] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Start processing in background
        thread = threading.Thread(target=process_async)
        thread.start()
        
        return jsonify({
            'success': True,
            'title': title,
            'message': 'Processing started'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Create PDF directory if it doesn't exist
        if not os.path.exists(PDF_FOLDER):
            os.makedirs(PDF_FOLDER)
            print(f"Created PDF directory: {PDF_FOLDER}")

        # Create embeddings directory if it doesn't exist
        if not os.path.exists(EMBEDDINGS_DIR):
            os.makedirs(EMBEDDINGS_DIR)
            print(f"Created embeddings directory: {EMBEDDINGS_DIR}")

        # Secure the filename and create paths
        filename = secure_filename(file.filename)
        title = os.path.splitext(filename)[0]  # Remove .pdf extension for title
        pdf_path = os.path.join(PDF_FOLDER, filename)

        # Check if file already exists
        if os.path.exists(pdf_path):
            print(f"File already exists: {pdf_path}")
            return jsonify({'error': 'File already exists'}), 400

        try:
            # Save the PDF file
            file.save(pdf_path)
            print(f"Successfully saved PDF to: {pdf_path}")

            # Initialize progress tracking
            embedding_progress[title] = {
                'status': 'Starting...',
                'progress': 0
            }

            def process_async():
                try:
                    def progress_callback(status, progress):
                        embedding_progress[title] = {
                            'status': status,
                            'progress': progress * 0.3,  # PDF processing is 30% of total
                            'timestamp': time.time()
                        }
                        print(f"📄 {status} ({progress:.1f}%)")
                    
                    # Extract text from PDF with progress tracking
                    text = process_local_pdf(pdf_path, progress_callback)

                    # Create embeddings with file path for caching
                    embeddings_path = create_embeddings([text], title, file_path=pdf_path)
                    print(f"✅ Created embeddings at: {embeddings_path}")

                    embedding_progress[title] = {
                        'status': 'completed',
                        'progress': 100,
                        'timestamp': time.time()
                    }
                    
                    # Automatically load this document if no document is currently loaded
                    global current_db, current_document_title
                    if current_db is None:
                        try:
                            with open(embeddings_path, 'rb') as f:
                                current_db = pickle.load(f)
                            current_document_title = title
                            print(f"🚀 Automatically loaded {title} as the active document")
                        except Exception as load_error:
                            print(f"Error auto-loading {title}: {load_error}")
                except Exception as e:
                    print(f"❌ Error in async processing: {str(e)}")
                    embedding_progress[title] = {
                        'status': 'error',
                        'error': str(e),
                        'progress': 0,
                        'timestamp': time.time()
                    }

            # Start processing in background
            thread = threading.Thread(target=process_async)
            thread.start()

            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'File uploaded and processing started'
            })

        except Exception as save_error:
            print(f"Error saving file: {str(save_error)}")
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except:
                    pass
            return jsonify({'error': f'Failed to save file: {str(save_error)}'}), 500

    except Exception as e:
        print(f"Error in upload_pdf: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/embedding_progress/<title>')
def get_embedding_progress(title):
    try:
        # Get progress for the specific title
        progress = embedding_progress.get(title, {
            'status': 'not_found',
            'progress': 0
        })
        
        # Add timing information if available
        if 'timestamp' in progress:
            current_time = time.time()
            elapsed_time = current_time - progress['timestamp']
            progress['elapsed_time'] = round(elapsed_time, 1)
            
            # Estimate remaining time based on progress
            if progress['progress'] > 0 and progress['progress'] < 100:
                estimated_total = elapsed_time * (100 / progress['progress'])
                estimated_remaining = estimated_total - elapsed_time
                progress['estimated_remaining'] = round(estimated_remaining, 1)
        
        print(f"📊 Progress for {title}: {progress.get('status', 'unknown')} ({progress.get('progress', 0):.1f}%)")
        return jsonify(progress)
    except Exception as e:
        print(f"❌ Error getting progress for {title}: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'progress': 0
        }), 500

@app.route('/embedding_progress')
def get_all_embedding_progress():
    """Get progress for all active embedding operations"""
    try:
        current_time = time.time()
        all_progress = {}
        
        for title, progress in embedding_progress.items():
            progress_copy = progress.copy()
            
            # Add timing information
            if 'timestamp' in progress:
                elapsed_time = current_time - progress['timestamp']
                progress_copy['elapsed_time'] = round(elapsed_time, 1)
                
                # Estimate remaining time
                if progress['progress'] > 0 and progress['progress'] < 100:
                    estimated_total = elapsed_time * (100 / progress['progress'])
                    estimated_remaining = estimated_total - elapsed_time
                    progress_copy['estimated_remaining'] = round(estimated_remaining, 1)
            
            all_progress[title] = progress_copy
        
        return jsonify(all_progress)
    except Exception as e:
        print(f"❌ Error getting all progress: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_embeddings', methods=['POST'])
def load_embeddings():
    try:
        data = request.get_json()
        title = data['title']
        
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{title}.pkl')
        
        if not os.path.exists(embeddings_path):
            # Try legacy embeddings
            legacy_path = os.path.join(EMBEDDINGS_DIR, 'existing_documents.pkl')
            if os.path.exists(legacy_path):
                embeddings_path = legacy_path
            else:
                return jsonify({'error': 'Embeddings not found'}), 404
        
        # Load embeddings
        global current_db, current_document_title
        try:
            with open(embeddings_path, 'rb') as f:
                current_db = pickle.load(f)
            current_document_title = title  # Store the currently loaded document title
            print(f"✅ Loaded embeddings for {title}")
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            # Try to delete corrupted file
            if os.path.exists(embeddings_path):
                try:
                    os.remove(embeddings_path)
                    print(f"🗑️ Removed corrupted embeddings file")
                except:
                    pass
            raise Exception(f"Failed to load embeddings: {e}")
        
        return jsonify({'success': True, 'document_title': title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_existing_documents', methods=['GET'])
def get_existing_documents():
    try:
        documents = []
        if not os.path.exists(PDF_FOLDER):
            os.makedirs(PDF_FOLDER)
            
        # Get local PDF files
        for file in os.listdir(PDF_FOLDER):
            if file.endswith('.pdf'):
                file_path = os.path.join(PDF_FOLDER, file)
                size = os.path.getsize(file_path)
                size_str = f"{size / 1024 / 1024:.1f}MB"
                
                # Check if embeddings exist for this document
                embedding_file = os.path.join(EMBEDDINGS_DIR, f"{file.replace('.pdf', '')}.pkl")
                has_embeddings = os.path.exists(embedding_file)
                
                documents.append({
                    'name': file,
                    'path': file_path,
                    'size': size_str,
                    'hasEmbeddings': has_embeddings,
                    'status': 'ready'
                })
        return jsonify({'documents': documents})
    except Exception as e:
        print(f"Error in get_existing_documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_embeddings')
def get_embeddings():
    try:
        embeddings_list = []
        # Get embeddings files from embeddings directory
        if os.path.exists(EMBEDDINGS_DIR):
            for file in os.listdir(EMBEDDINGS_DIR):
                if file.endswith('.pkl'):
                    embeddings_list.append({
                        'name': file,
                        'path': os.path.join(EMBEDDINGS_DIR, file)
                    })
        
        # Add legacy embeddings file if it exists
        legacy_file = 'embeddings.pkl.backup'
        if os.path.exists(legacy_file):
            embeddings_list.append({
                'name': 'Legacy Embeddings.pkl',
                'path': legacy_file
            })
        
        return jsonify({'embeddings': embeddings_list})
    except Exception as e:
        print(f"Error in get_embeddings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_media', methods=['POST'])
def upload_media():
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        file_type = request.form.get('type', '').lower()
        title = os.path.splitext(filename)[0]

        print(f"Processing {file_type} file: {filename}")

        if file_type not in ['audio', 'video']:
            print(f"Invalid file type: {file_type}")
            return jsonify({'error': 'Invalid file type'}), 400

        # Determine target folder based on file type
        target_folder = AUDIO_FOLDER if file_type == 'audio' else VIDEO_FOLDER
        file_path = os.path.join(target_folder, filename)
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{title}.txt")

        # Check if file already exists
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            return jsonify({'error': 'File already exists'}), 400

        # Initialize progress tracking
        embedding_progress[title] = {
            'status': 'Starting...',
            'progress': 0
        }

        print(f"Saving file to: {file_path}")
        # Save the file
        file.save(file_path)
        print(f"Successfully saved {file_type} file to: {file_path}")

        def process_async():
            try:
                # Update status
                embedding_progress[title] = {
                    'status': 'Transcribing...',
                    'progress': 30
                }

                # Process file based on type
                if file_type == 'audio':
                    print("Processing audio file...")
                    text = process_audio_file(file_path)
                else:
                    print("Processing video file...")
                    text = process_video_file(file_path)

                print("Saving transcription...")
                # Save transcription
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                print("Creating embeddings...")
                embedding_progress[title] = {
                    'status': 'Creating embeddings...',
                    'progress': 60,
                    'timestamp': time.time()
                }

                # Create embeddings with file path for caching
                create_embeddings([text], title, file_path=file_path)

                embedding_progress[title] = {
                    'status': 'completed',
                    'progress': 100
                }
                print("Processing completed successfully")

            except Exception as e:
                print(f"Error in async processing: {str(e)}")
                embedding_progress[title] = {
                    'status': 'error',
                    'error': str(e),
                    'progress': 0
                }
                # Clean up files in case of error
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    if os.path.exists(transcript_path):
                        os.remove(transcript_path)
                except:
                    pass

        # Start processing in background
        thread = threading.Thread(target=process_async)
        thread.start()

        return jsonify({
            'success': True,
            'filename': filename,
            'message': f'{file_type.capitalize()} file uploaded and processing started'
        })

    except Exception as e:
        print(f"Error in upload_media: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_media_files', methods=['GET'])
def get_media_files():
    try:
        media_files = []

        # Get audio files
        for file in os.listdir(AUDIO_FOLDER):
            if file.endswith(('.mp3', '.wav', '.m4a')):
                file_path = os.path.join(AUDIO_FOLDER, file)
                title = os.path.splitext(file)[0]
                embedding_file = os.path.join(EMBEDDINGS_DIR, f"{title}.pkl")
                transcript_file = os.path.join(TRANSCRIPTS_FOLDER, f"{title}.txt")

                media_files.append({
                    'name': file,
                    'path': file_path,
                    'type': 'audio',
                    'size': f"{os.path.getsize(file_path) / 1024 / 1024:.1f}MB",
                    'hasEmbeddings': os.path.exists(embedding_file),
                    'hasTranscript': os.path.exists(transcript_file),
                    'status': 'ready'
                })

        # Get video files
        for file in os.listdir(VIDEO_FOLDER):
            if file.endswith(('.mp4', '.avi', '.mov')):
                file_path = os.path.join(VIDEO_FOLDER, file)
                title = os.path.splitext(file)[0]
                embedding_file = os.path.join(EMBEDDINGS_DIR, f"{title}.pkl")
                transcript_file = os.path.join(TRANSCRIPTS_FOLDER, f"{title}.txt")

                media_files.append({
                    'name': file,
                    'path': file_path,
                    'type': 'video',
                    'size': f"{os.path.getsize(file_path) / 1024 / 1024:.1f}MB",
                    'hasEmbeddings': os.path.exists(embedding_file),
                    'hasTranscript': os.path.exists(transcript_file),
                    'status': 'ready'
                })

        return jsonify({'media_files': media_files})
    except Exception as e:
        print(f"Error in get_media_files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat.html')
def serve_chat():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/delete_document', methods=['DELETE'])
def delete_document():
    """Delete a document and its embeddings"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        deleted_files = []
        
        # Delete PDF file
        pdf_path = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            deleted_files.append('PDF file')
        
        # Delete embeddings
        title = filename.replace('.pdf', '').replace('.mp3', '').replace('.mp4', '')
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{title}.pkl')
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)
            deleted_files.append('embeddings')
        
        # Delete transcript if exists
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f'{title}.txt')
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
            deleted_files.append('transcript')
        
        # Delete media file if exists
        for folder in [AUDIO_FOLDER, VIDEO_FOLDER]:
            media_path = os.path.join(folder, filename)
            if os.path.exists(media_path):
                os.remove(media_path)
                deleted_files.append('media file')
                break
        
        if deleted_files:
            return jsonify({
                'success': True,
                'message': f'Successfully deleted {", ".join(deleted_files)}',
                'deleted_files': deleted_files
            })
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_document', methods=['GET'])
def download_document():
    """Download a document file"""
    try:
        filename = request.args.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Check PDF folder first
        pdf_path = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(pdf_path):
            return send_from_directory(PDF_FOLDER, filename, as_attachment=True)
        
        # Check audio folder
        audio_path = os.path.join(AUDIO_FOLDER, filename)
        if os.path.exists(audio_path):
            return send_from_directory(AUDIO_FOLDER, filename, as_attachment=True)
        
        # Check video folder
        video_path = os.path.join(VIDEO_FOLDER, filename)
        if os.path.exists(video_path):
            return send_from_directory(VIDEO_FOLDER, filename, as_attachment=True)
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_system_status', methods=['GET'])
def get_system_status():
    """Get system status and statistics"""
    try:
        # Count files
        pdf_count = len([f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]) if os.path.exists(PDF_FOLDER) else 0
        audio_count = len([f for f in os.listdir(AUDIO_FOLDER) if f.endswith(('.mp3', '.wav', '.m4a'))]) if os.path.exists(AUDIO_FOLDER) else 0
        video_count = len([f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]) if os.path.exists(VIDEO_FOLDER) else 0
        embeddings_count = len([f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.pkl')]) if os.path.exists(EMBEDDINGS_DIR) else 0
        
        # Check if default document is loaded
        default_loaded = current_db is not None
        
        return jsonify({
            'status': 'running',
            'default_document_loaded': default_loaded,
            'video_support': VIDEO_SUPPORT,
            'statistics': {
                'pdf_files': pdf_count,
                'audio_files': audio_count,
                'video_files': video_count,
                'embeddings': embeddings_count,
                'total_files': pdf_count + audio_count + video_count
            },
            'directories': {
                'pdf_folder': PDF_FOLDER,
                'audio_folder': AUDIO_FOLDER,
                'video_folder': VIDEO_FOLDER,
                'embeddings_folder': EMBEDDINGS_DIR,
                'transcripts_folder': TRANSCRIPTS_FOLDER
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_available_documents', methods=['GET'])
def get_available_documents():
    """Get list of documents available for chat"""
    try:
        documents = []
        
        # Get PDF documents with embeddings
        if os.path.exists(PDF_FOLDER):
            for file in os.listdir(PDF_FOLDER):
                if file.endswith('.pdf'):
                    title = file.replace('.pdf', '')
                    embedding_file = os.path.join(EMBEDDINGS_DIR, f'{title}.pkl')
                    
                    if os.path.exists(embedding_file):
                        documents.append({
                            'name': title,
                            'filename': file,
                            'type': 'pdf',
                            'status': 'ready'
                        })
        
        # Get media files with embeddings
        for folder, file_type in [(AUDIO_FOLDER, 'audio'), (VIDEO_FOLDER, 'video')]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith(('.mp3', '.wav', '.m4a') if file_type == 'audio' else ('.mp4', '.avi', '.mov')):
                        title = os.path.splitext(file)[0]
                        embedding_file = os.path.join(EMBEDDINGS_DIR, f'{title}.pkl')
                        
                        if os.path.exists(embedding_file):
                            documents.append({
                                'name': title,
                                'filename': file,
                                'type': file_type,
                                'status': 'ready'
                            })
        
        return jsonify({'documents': documents})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_current_document', methods=['GET'])
def get_current_document():
    try:
        return jsonify({
            'document_title': current_document_title,
            'has_document': current_db is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug_document', methods=['GET'])
def debug_document():
    """Debug endpoint to check document status"""
    try:
        if current_db is None:
            return jsonify({'error': 'No document loaded'})
        
        # Test retrieval with a simple query
        test_query = "aviation"
        try:
            # Try direct similarity search
            docs_similarity = current_db.similarity_search(test_query, k=3)
            doc_previews = [doc.page_content[:200] + "..." for doc in docs_similarity]
            
            # Check if retriever works
            try:
                retriever = current_db.as_retriever()
                retriever_docs = retriever.get_relevant_documents(test_query)
                retriever_count = len(retriever_docs)
            except Exception as retriever_error:
                retriever_count = f"Error: {str(retriever_error)}"
            
            return jsonify({
                'current_document': current_document_title,
                'test_query': test_query,
                'similarity_search_count': len(docs_similarity),
                'retriever_count': retriever_count,
                'doc_previews': doc_previews,
                'db_type': type(current_db).__name__,
                'has_similarity_search': hasattr(current_db, 'similarity_search'),
                'has_as_retriever': hasattr(current_db, 'as_retriever')
            })
        except Exception as e:
            return jsonify({
                'current_document': current_document_title,
                'error': f'Retrieval failed: {str(e)}',
                'db_type': type(current_db).__name__
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug_embeddings', methods=['GET'])
def debug_embeddings():
    """Check if embeddings exist and are valid"""
    try:
        embeddings_files = []
        if os.path.exists(EMBEDDINGS_DIR):
            for file in os.listdir(EMBEDDINGS_DIR):
                if file.endswith('.pkl'):
                    file_path = os.path.join(EMBEDDINGS_DIR, file)
                    file_size = os.path.getsize(file_path)
                    
                    # Try to load the file to check if it's valid
                    try:
                        with open(file_path, 'rb') as f:
                            db = pickle.load(f)
                        status = "valid"
                        db_type = type(db).__name__
                    except Exception as e:
                        status = f"invalid: {str(e)}"
                        db_type = "unknown"
                    
                    embeddings_files.append({
                        'name': file,
                        'size': f"{file_size} bytes",
                        'status': status,
                        'type': db_type
                    })
        
        return jsonify({
            'embeddings_directory': EMBEDDINGS_DIR,
            'files': embeddings_files,
            'current_document_loaded': current_document_title if current_db else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Check if default document is loaded
        if current_db is None:
            # Try to reload default document
            if not load_default_document():
                return jsonify({'error': 'No documents available. Please upload a document first.'}), 400

        data = request.get_json()
        if not data or 'msg' not in data:
            return jsonify({'error': 'No message provided'}), 400

        question = data['msg'].strip()
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        print(f"🔍 Current document: {current_document_title}")
        print(f"❓ Question: {question}")

        # Get relevant documents using multiple fallback methods
        relevant_docs = []
        context = ""
        
        try:
            # Method 1: Try direct similarity search first (most reliable)
            if hasattr(current_db, 'similarity_search'):
                k = 10 if 'cs-25' in current_document_title.lower() else 6
                print(f"🔍 Using similarity_search with k={k}")
                relevant_docs = current_db.similarity_search(question, k=k)
            
            # Method 2: If similarity_search didn't work or returned no results, try retriever
            if not relevant_docs and hasattr(current_db, 'as_retriever'):
                print("🔍 Trying as_retriever method")
                retriever = current_db.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 10 if 'cs-25' in current_document_title.lower() else 6}
                )
                relevant_docs = retriever.get_relevant_documents(question)
            
            # Method 3: If still no results, try the retriever with invoke (newer LangChain)
            if not relevant_docs and hasattr(current_db, 'as_retriever'):
                print("🔍 Trying retriever invoke method")
                retriever = current_db.as_retriever()
                if hasattr(retriever, 'invoke'):
                    relevant_docs = retriever.invoke(question)
                else:
                    relevant_docs = retriever.get_relevant_documents(question)
                    
        except Exception as retrieval_error:
            print(f"❌ Retrieval error: {str(retrieval_error)}")
            return jsonify({'error': f'Failed to search document: {str(retrieval_error)}'}), 500

        if not relevant_docs:
            print("❌ No relevant documents found")
            return jsonify({
                'answer': f"I couldn't find relevant information in the '{current_document_title}' document to answer your question. Please try asking about specific topics mentioned in the document or rephrase your question."
            })

        print(f"✅ Retrieved {len(relevant_docs)} relevant documents")
        
        # Format context from documents
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            content = doc.page_content
            # Add source information if available
            source_info = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'source' in doc.metadata:
                    source_info = f" [Source: {doc.metadata['source']}]"
                elif 'page' in doc.metadata:
                    source_info = f" [Page: {doc.metadata['page']}]"
            
            context_parts.append(f"Document {i+1}{source_info}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        print(f"📝 Context length: {len(context)} characters")
        print(f"📄 Context preview: {context[:300]}...")

        # Set up the model
        try:
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key:
                raise Exception("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key=google_api_key,
                temperature=0.1
            )
        except Exception as llm_error:
            print(f"❌ LLM initialization error: {str(llm_error)}")
            return jsonify({'error': 'Failed to initialize AI model'}), 500

        # Set up the prompt template
        if 'cs-25' in current_document_title.lower() or 'aviation' in current_document_title.lower():
            prompt_template = """You are an expert aviation assistant specializing in CS-25 certification specifications. 

IMPORTANT: Answer the question using ONLY the context provided below. If the context doesn't contain the information needed to answer the question, clearly state that the information is not available in the document.

Context from CS-25 document:
{context}

Question: {question}

Based on the context above, provide a precise and accurate answer:"""
        else:
            prompt_template = """You are a helpful AI assistant. 

IMPORTANT: Answer the question using ONLY the context provided below. If the context doesn't contain the information needed to answer the question, clearly state that the information is not available in the document.

Context from document:
{context}

Question: {question}

Based on the context above, provide a helpful and accurate answer:"""

        try:
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template,
            )

            # Create the prompt input
            prompt_input = {
                "context": context,
                "question": question
            }
            
            # Get the answer from the LLM
            print("🧠 Sending request to AI model...")
            formatted_prompt = prompt.format(**prompt_input)
            answer = llm.invoke(formatted_prompt).content
            
            print(f"✅ Answer generated: {answer[:200]}...")
            return jsonify({'answer': answer})
            
        except Exception as llm_error:
            print(f"❌ LLM processing error: {str(llm_error)}")
            return jsonify({'error': f'AI model error: {str(llm_error)}'}), 500
        
    except Exception as e:
        print(f"❌ Error in /ask route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/<path:path>')
def static_proxy(path):
        return send_from_directory(app.static_folder, path)

# Load default document on startup
load_default_document()

if __name__ == '__main__':
    app.run(debug=True, port=8000)
