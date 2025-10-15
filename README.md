# Aviation Document Assistant

A sophisticated RAG (Retrieval-Augmented Generation) chatbot system specifically designed for aviation document analysis, with special focus on CS-25 certification specifications.

## ✈️ Features

- **Multi-format Document Processing**: PDFs, audio files, and video files
- **Advanced NLP Pipeline**: Text extraction, chunking, and embedding generation
- **Intelligent Chat Interface**: Powered by Google Gemini AI for aviation-specific queries
- **Real-time Processing**: With progress tracking and status updates
- **Document Management**: Upload, organize, and manage various document types
- **Semantic Search**: Real embeddings for accurate content retrieval

## 🏗️ Architecture

- **Backend**: Flask-based Python server with comprehensive API endpoints
- **Frontend**: Modern HTML5/CSS3/JavaScript interface
- **AI Integration**: LangChain, FAISS vector database, Google Gemini AI
- **File Processing**: PDF extraction, audio/video transcription with Whisper
- **Embedding System**: Sentence transformers and FAISS for semantic search

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd aviation_bot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv bot
   bot\Scripts\activate  # Windows
   # or
   source bot/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up directories**
   ```bash
   mkdir PDFS AUDIO VIDEO embeddings transcripts uploads
   ```

5. **Run the application**
   ```bash
   python faiss-chatbot.py
   ```

6. **Access the web interface**
   - Open your browser and go to `http://localhost:8000`

## 📁 Project Structure

```
aviation_bot/
├── faiss-chatbot.py          # Main Flask application
├── ragchatbot.py             # Alternative RAG implementation
├── requirements.txt          # Python dependencies
├── client/build/             # Frontend files
│   ├── index.html           # Main chat interface
│   ├── styles.css           # Styling
│   ├── script.js            # Chat functionality
│   └── documents.js         # Document management
├── PDFS/                    # PDF documents
├── AUDIO/                   # Audio files
├── VIDEO/                   # Video files
├── embeddings/              # Generated embeddings
├── transcripts/             # Audio/video transcripts
└── uploads/                 # Temporary uploads
```

## 🔧 Configuration

### Environment Variables

The system uses environment variables for secure API key management.

1. **Create a `.env` file** in the project root:
   ```bash
   cp env.example .env
   ```

2. **Edit the `.env` file** and add your API keys:
   ```env
   # Google Gemini API Key
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Get your Google Gemini API key** from: https://makersuite.google.com/app/apikey

**Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

### Embedding Models

The system supports multiple embedding methods:

1. **Sentence Transformers** (Primary): `all-MiniLM-L6-v2`
2. **Ollama** (Fallback): `all-minilm`
3. **Enhanced Instant** (Final fallback): Custom implementation

## 📚 Usage

### Uploading Documents

1. **PDF Documents**: Upload via the web interface or place in `PDFS/` directory
2. **Audio/Video**: Upload via the web interface for automatic transcription
3. **Processing**: Documents are automatically processed and embedded

### Chat Interface

1. **Select Document**: Choose from available documents in the sidebar
2. **Ask Questions**: Type aviation-related questions
3. **Get Answers**: Receive accurate, context-aware responses

### API Endpoints

- `POST /upload_pdf` - Upload PDF documents
- `POST /upload_media` - Upload audio/video files
- `POST /ask` - Ask questions about documents
- `GET /debug_document` - Debug document status
- `GET /debug_embeddings` - Check embedding files

## 🎯 Special Features

### Aviation-Specific Processing

- **CS-25 Document Handling**: Specialized processing for aviation certification documents
- **Technical Terminology**: Optimized for aviation and regulatory language
- **Section Recognition**: Intelligent chunking based on CS-25 structure

### Real-time Progress Tracking

- **Upload Progress**: Real-time feedback during file uploads
- **Processing Status**: Live updates during embedding generation
- **Error Handling**: Comprehensive error reporting and recovery

## 🔍 Debugging

### Debug Endpoints

- `/debug_document` - Check current document status
- `/debug_embeddings` - Verify embedding files
- `/get_system_status` - System health check

### Common Issues

1. **Memory Issues**: Reduce batch size in embedding generation
2. **Model Loading**: Check internet connection for model downloads
3. **File Permissions**: Ensure write access to directories

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **FAISS** for vector similarity search
- **Sentence Transformers** for embeddings
- **Google Gemini** for language model
- **Whisper** for audio transcription

## 📞 Support

For issues and questions:
1. Check the debugging endpoints
2. Review the logs in the terminal
3. Open an issue on GitHub

---

**Built with ❤️ for the aviation community**
