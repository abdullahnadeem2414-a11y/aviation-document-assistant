// Aviation Document Assistant - Main Chat Interface
// Global state management
let currentDocument = null;
let chatHistory = [];
let isProcessing = false;
let currentDocumentTitle = null;

// DOM elements
const chatContainer = document.getElementById('chatContainer');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const statusIndicator = document.getElementById('statusIndicator');
const documentsList = document.getElementById('documentsList');
const mediaList = document.getElementById('mediaList');
const uploadModal = document.getElementById('uploadModal');
const closeModal = document.getElementById('closeModal');
const pdfUploadArea = document.getElementById('pdfUploadArea');
const mediaUploadArea = document.getElementById('mediaUploadArea');
const pdfInput = document.getElementById('pdfInput');
const mediaInput = document.getElementById('mediaInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadAvailableDocuments();
    checkCurrentDocument();
    loadChatHistory();
});

function initializeApp() {
    console.log('🚀 Aviation Document Assistant initialized');
    updateStatus('Ready', 'ready');
    setupAutoResize();
    setupKeyboardShortcuts();
}

function setupEventListeners() {
    // Chat input and send button
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', handleKeyDown);
    chatInput.addEventListener('input', handleInputChange);
    
    // Modal controls
    closeModal.addEventListener('click', hideUploadModal);
    uploadModal.addEventListener('click', (e) => {
        if (e.target === uploadModal) hideUploadModal();
    });
    
    // Upload areas
    pdfUploadArea.addEventListener('click', () => pdfInput.click());
    mediaUploadArea.addEventListener('click', () => mediaInput.click());
    
    // File inputs
    pdfInput.addEventListener('change', handlePdfUpload);
    mediaInput.addEventListener('change', handleMediaUpload);
    
    // Drag and drop
    setupDragAndDrop();
    
    // Navigation
    setupNavigation();
}

function setupAutoResize() {
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to send message
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            hideUploadModal();
        }
    });
}

function setupNavigation() {
    // Handle navigation between pages
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            // Remove active class from all nav buttons
            navButtons.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
        });
    });
}

function setupDragAndDrop() {
    [pdfUploadArea, mediaUploadArea].forEach(area => {
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.style.borderColor = '#10a37f';
            area.style.backgroundColor = '#3d3d3d';
        });
        
        area.addEventListener('dragleave', (e) => {
            e.preventDefault();
            area.style.borderColor = '#3d3d3d';
            area.style.backgroundColor = 'transparent';
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.style.borderColor = '#3d3d3d';
            area.style.backgroundColor = 'transparent';
            
            const files = Array.from(e.dataTransfer.files);
            if (area === pdfUploadArea) {
                handlePdfFiles(files);
            } else {
                handleMediaFiles(files);
            }
        });
    });
}

// Chat functionality
function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function handleInputChange() {
    const hasText = chatInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isProcessing;
    chatInput.disabled = isProcessing;
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isProcessing) return;
    
    // Check if document is loaded
    if (!currentDocument) {
        showNotification('Please select a document first', 'warning');
        return;
    }
    
    // Add user message to chat
    addMessageToChat('user', message);
    chatInput.value = '';
    handleInputChange();
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        isProcessing = true;
        updateStatus('Processing...', 'processing');
        
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ msg: message })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Add assistant response to chat
        addMessageToChat('assistant', data.answer);
        
        // Save to chat history
        saveChatHistory();
        
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToChat('assistant', `Sorry, I encountered an error: ${error.message}`);
        showNotification('Error sending message', 'error');
    } finally {
        hideTypingIndicator();
        isProcessing = false;
        updateStatus('Ready', 'ready');
    }
}

function addMessageToChat(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (role === 'user') {
        messageContent.textContent = content;
    } else {
        // Format assistant response with markdown-like formatting
        messageContent.innerHTML = formatAssistantResponse(content);
    }
    
    messageDiv.appendChild(messageContent);
    
    // Add message actions
    const messageActions = document.createElement('div');
    messageActions.className = 'message-actions';
    messageActions.innerHTML = `
        <button class="message-action" onclick="copyMessage(this)" title="Copy">
            <i class="fas fa-copy"></i>
        </button>
        <button class="message-action" onclick="regenerateResponse(this)" title="Regenerate">
            <i class="fas fa-redo"></i>
        </button>
    `;
    
    messageDiv.appendChild(messageActions);
    
    // Hide welcome message and add to chat
    hideWelcomeMessage();
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatAssistantResponse(content) {
    // Basic markdown-like formatting
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    hideWelcomeMessage();
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function hideWelcomeMessage() {
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
}

function showWelcomeMessage() {
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'flex';
    }
}

// Document management
async function loadAvailableDocuments() {
    try {
        const response = await fetch('/get_available_documents');
        const data = await response.json();
        
        if (data.documents) {
            renderDocumentsList(data.documents);
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function renderDocumentsList(documents) {
    documentsList.innerHTML = '';
    
    if (documents.length === 0) {
        documentsList.innerHTML = `
            <div class="empty-documents">
                <i class="fas fa-folder-open"></i>
                <p>No documents available</p>
                <button class="upload-btn" onclick="showUploadModal()">
                    <i class="fas fa-plus"></i>
                    Upload Document
                </button>
            </div>
        `;
        return;
    }
    
    documents.forEach(doc => {
        const docElement = createDocumentElement(doc);
        documentsList.appendChild(docElement);
    });
}

function createDocumentElement(doc) {
    const div = document.createElement('div');
    div.className = `document-item ${currentDocumentTitle === doc.name ? 'active' : ''}`;
    div.dataset.documentName = doc.name;
    
    const icon = getDocumentIcon(doc.type);
    
    div.innerHTML = `
        <div class="file-info">
            <div class="file-row">
                <div class="file-icon">
                    <i class="${icon}"></i>
                </div>
                <div class="file-name" title="${doc.name}">${doc.name}</div>
            </div>
            <div class="file-status">Ready for chat</div>
        </div>
    `;
    
    div.addEventListener('click', () => loadDocument(doc.name));
    
    return div;
}

function getDocumentIcon(type) {
    switch (type) {
        case 'pdf':
            return 'fas fa-file-pdf';
        case 'audio':
            return 'fas fa-file-audio';
        case 'video':
            return 'fas fa-file-video';
        default:
            return 'fas fa-file';
    }
}

async function loadDocument(documentName) {
    try {
        showNotification('Loading document...', 'info');
        
        const response = await fetch('/load_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: documentName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentDocument = true;
            currentDocumentTitle = documentName;
            
            // Update UI
            updateDocumentSelection(documentName);
            updateStatus(`Loaded: ${documentName}`, 'loaded');
            enableChatInput();
            
            // Update chat header
            updateChatHeader(documentName);
            
            showNotification(`Document "${documentName}" loaded successfully!`, 'success');
            
            // Save to localStorage
            localStorage.setItem('currentDocument', documentName);
            
        } else {
            throw new Error(data.error || 'Failed to load document');
        }
        
    } catch (error) {
        console.error('Error loading document:', error);
        showNotification(`Error loading document: ${error.message}`, 'error');
    }
}

function updateDocumentSelection(selectedName) {
    // Update document list
    document.querySelectorAll('.document-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.documentName === selectedName) {
            item.classList.add('active');
        }
    });
}

function updateChatHeader(documentName) {
    const chatTitle = document.querySelector('.chat-title h2');
    if (chatTitle) {
        chatTitle.textContent = `Aviation Assistant - ${documentName}`;
    }
}

function enableChatInput() {
    chatInput.disabled = false;
    chatInput.placeholder = 'Ask anything about aviation...';
    sendBtn.disabled = false;
}

function disableChatInput() {
    chatInput.disabled = true;
    chatInput.placeholder = 'Select a document to start chatting...';
    sendBtn.disabled = true;
}

async function checkCurrentDocument() {
    try {
        const response = await fetch('/get_current_document');
        const data = await response.json();
        
        if (data.has_document && data.document_title) {
            currentDocument = true;
            currentDocumentTitle = data.document_title;
            updateDocumentSelection(data.document_title);
            updateStatus(`Loaded: ${data.document_title}`, 'loaded');
            enableChatInput();
            updateChatHeader(data.document_title);
        } else {
            disableChatInput();
        }
    } catch (error) {
        console.error('Error checking current document:', error);
        disableChatInput();
    }
}

// Upload functionality
function showUploadModal() {
    uploadModal.classList.add('show');
    resetUploadForm();
}

function hideUploadModal() {
    uploadModal.classList.remove('show');
    resetUploadForm();
}

function resetUploadForm() {
    pdfInput.value = '';
    mediaInput.value = '';
    uploadProgress.style.display = 'none';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading...';
}

function handlePdfUpload(e) {
    const files = Array.from(e.target.files);
    handlePdfFiles(files);
}

function handleMediaUpload(e) {
    const files = Array.from(e.target.files);
    handleMediaFiles(files);
}

function handlePdfFiles(files) {
    const pdfFiles = files.filter(file => file.type === 'application/pdf');
    if (pdfFiles.length === 0) {
        showNotification('Please select PDF files only.', 'error');
        return;
    }
    
    uploadFiles(pdfFiles, 'pdf');
}

function handleMediaFiles(files) {
    const mediaFiles = files.filter(file => 
        file.type.startsWith('audio/') || file.type.startsWith('video/')
    );
    if (mediaFiles.length === 0) {
        showNotification('Please select audio or video files only.', 'error');
        return;
    }
    
    uploadFiles(mediaFiles, 'media');
}

async function uploadFiles(files, type) {
    uploadProgress.style.display = 'block';
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const progress = ((i + 1) / files.length) * 100;
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Uploading ${file.name}...`;
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            let endpoint = '/upload_pdf';
            if (type === 'media') {
                endpoint = '/upload_media';
                formData.append('type', file.type.startsWith('audio/') ? 'audio' : 'video');
            }
            
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            showNotification(`Successfully uploaded ${file.name}`, 'success');
            
        } catch (error) {
            console.error(`Error uploading ${file.name}:`, error);
            showNotification(`Error uploading ${file.name}: ${error.message}`, 'error');
        }
    }
    
    // Refresh document list
    setTimeout(() => {
        loadAvailableDocuments();
        hideUploadModal();
    }, 1000);
}

// Utility functions
function updateStatus(message, type) {
    statusIndicator.textContent = message;
    statusIndicator.className = `status ${type}`;
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

function copyMessage(button) {
    const messageContent = button.closest('.message').querySelector('.message-content');
    const text = messageContent.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Message copied to clipboard', 'success');
    }).catch(() => {
        showNotification('Failed to copy message', 'error');
    });
}

function regenerateResponse(button) {
    const messageElement = button.closest('.message');
    const userMessage = getLastUserMessage();
    
    if (userMessage) {
        // Remove the current assistant response
        messageElement.remove();
        
        // Send the message again
        chatInput.value = userMessage;
        sendMessage();
    }
}

function getLastUserMessage() {
    const messages = document.querySelectorAll('.message.user');
    if (messages.length > 0) {
        return messages[messages.length - 1].querySelector('.message-content').textContent;
    }
    return null;
}

// Chat history management
function saveChatHistory() {
    const messages = document.querySelectorAll('.message');
    const history = [];
    
    messages.forEach(message => {
        const role = message.classList.contains('user') ? 'user' : 'assistant';
        const content = message.querySelector('.message-content').textContent;
        history.push({ role, content });
    });
    
    localStorage.setItem('chatHistory', JSON.stringify(history));
}

function loadChatHistory() {
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
        try {
            const history = JSON.parse(savedHistory);
            history.forEach(msg => {
                addMessageToChat(msg.role, msg.content);
            });
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
}

function clearChatHistory() {
    chatHistory = [];
    localStorage.removeItem('chatHistory');
    
    // Clear chat container
    const messages = document.querySelectorAll('.message');
    messages.forEach(msg => msg.remove());
    
    showWelcomeMessage();
    showNotification('Chat history cleared', 'info');
}

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
        padding: 16px 20px;
        color: #ffffff;
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification.success {
        border-color: #10a37f;
        background-color: #10a37f20;
    }
    
    .notification.error {
        border-color: #dc2626;
        background-color: #dc262620;
    }
    
    .notification.warning {
        border-color: #f59e0b;
        background-color: #f59e0b20;
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .notification-content i {
        font-size: 16px;
    }
    
    .notification.success i {
        color: #10a37f;
    }
    
    .notification.error i {
        color: #dc2626;
    }
    
    .notification.warning i {
        color: #f59e0b;
    }
    
    .typing-indicator {
        opacity: 0.7;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
        align-items: center;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #8e8ea0;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .empty-documents {
        text-align: center;
        padding: 40px 20px;
        color: #8e8ea0;
    }
    
    .empty-documents i {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
    }
    
    .empty-documents p {
        margin-bottom: 20px;
        font-size: 16px;
    }
    
    .upload-btn {
        background-color: #10a37f;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .upload-btn:hover {
        background-color: #0d8a6b;
        transform: translateY(-1px);
    }
    
    .status.processing {
        color: #f59e0b;
    }
    
    .status.loaded {
        color: #10a37f;
    }
    
    .status.error {
        color: #dc2626;
    }
`;

// Add styles to page
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Export functions for global access
window.copyMessage = copyMessage;
window.regenerateResponse = regenerateResponse;
window.showUploadModal = showUploadModal;
window.loadDocument = loadDocument;