// Global state
let allFiles = [];
let filteredFiles = [];
let currentFilter = 'all';
let selectedFiles = new Set();
let isUploading = false;

// DOM elements
const filesList = document.getElementById('filesList');
const emptyState = document.getElementById('emptyState');
const uploadModal = document.getElementById('uploadModal');
const uploadBtn = document.getElementById('uploadBtn');
const closeModal = document.getElementById('closeModal');
const refreshBtn = document.getElementById('refreshBtn');
const viewToggle = document.getElementById('viewToggle');
const fileDetailsModal = document.getElementById('fileDetailsModal');
const closeDetailsModal = document.getElementById('closeDetailsModal');

// Statistics elements
const pdfCount = document.getElementById('pdfCount');
const audioCount = document.getElementById('audioCount');
const videoCount = document.getElementById('videoCount');
const processedCount = document.getElementById('processedCount');
const filesCount = document.getElementById('filesCount');

// Upload elements
const pdfUploadArea = document.getElementById('pdfUploadArea');
const mediaUploadArea = document.getElementById('mediaUploadArea');
const batchUploadArea = document.getElementById('batchUploadArea');
const pdfInput = document.getElementById('pdfInput');
const mediaInput = document.getElementById('mediaInput');
const batchInput = document.getElementById('batchInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressPercentage = document.getElementById('progressPercentage');
const progressDetails = document.getElementById('progressDetails');

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadAllFiles();
    // Start progress tracking for any files currently processing
    startProgressTracking();
    // Also check for completed files that might need status refresh
    setTimeout(refreshCompletedFiles, 2000);
});

function initializeApp() {
    updateStatistics();
    updateFileCount();
}

function setupEventListeners() {
    // Modal controls
    uploadBtn.addEventListener('click', showUploadModal);
    closeModal.addEventListener('click', hideUploadModal);
    
    // File management
    refreshBtn.addEventListener('click', loadAllFiles);
    viewToggle.addEventListener('click', toggleView);
    
    // Tab switching
    document.querySelectorAll('.file-tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => switchTab(e.target.dataset.tab));
    });
    
    // Upload tabs
    document.querySelectorAll('.upload-tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => switchUploadTab(e.target.dataset.tab));
    });
    
    // File uploads
    pdfUploadArea.addEventListener('click', () => pdfInput.click());
    mediaUploadArea.addEventListener('click', () => mediaInput.click());
    batchUploadArea.addEventListener('click', () => batchInput.click());
    pdfInput.addEventListener('change', handlePdfUpload);
    mediaInput.addEventListener('change', handleMediaUpload);
    batchInput.addEventListener('change', handleBatchUpload);
    
    // Drag and drop
    setupDragAndDrop();
    
    // Clear logs button
    const clearLogsBtn = document.getElementById('clearLogsBtn');
    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', clearLogs);
    }
    
    // Refresh files button
    const refreshFilesBtn = document.getElementById('refreshFilesBtn');
    if (refreshFilesBtn) {
        refreshFilesBtn.addEventListener('click', refreshCompletedFiles);
    }
    
    // Modal close on outside click
    uploadModal.addEventListener('click', (e) => {
        if (e.target === uploadModal) hideUploadModal();
    });
    
    fileDetailsModal.addEventListener('click', (e) => {
        if (e.target === fileDetailsModal) hideFileDetailsModal();
    });
    
    closeDetailsModal.addEventListener('click', hideFileDetailsModal);
}

function switchTab(tabName) {
    currentFilter = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.file-tabs .tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Filter files
    filterFiles();
    renderFiles();
}

function switchUploadTab(tabName) {
    // Update upload tab buttons
    document.querySelectorAll('.upload-tabs .tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}Tab`).classList.add('active');
}

async function loadAllFiles() {
    try {
        // Load PDF documents
        const pdfResponse = await fetch('/get_existing_documents');
        const pdfData = await pdfResponse.json();
        
        // Load media files
        const mediaResponse = await fetch('/get_media_files');
        const mediaData = await mediaResponse.json();
        
        // Combine all files
        allFiles = [];
        
        if (pdfData.documents) {
            pdfData.documents.forEach(doc => {
                allFiles.push({
                    ...doc,
                    type: 'pdf',
                    category: 'document'
                });
            });
        }
        
        if (mediaData.media_files) {
            mediaData.media_files.forEach(file => {
                allFiles.push({
                    ...file,
                    type: file.type,
                    category: 'media'
                });
            });
        }
        
        // Sort by upload date (newest first)
        allFiles.sort((a, b) => new Date(b.uploadDate || 0) - new Date(a.uploadDate || 0));
        
        filterFiles();
        renderFiles();
        updateStatistics();
        updateFileCount();
        
    } catch (error) {
        console.error('Error loading files:', error);
        showNotification('Error loading files', 'error');
    }
}

function filterFiles() {
    switch (currentFilter) {
        case 'pdfs':
            filteredFiles = allFiles.filter(file => file.type === 'pdf');
            break;
        case 'audio':
            filteredFiles = allFiles.filter(file => file.type === 'audio');
            break;
        case 'video':
            filteredFiles = allFiles.filter(file => file.type === 'video');
            break;
        default:
            filteredFiles = [...allFiles];
    }
}

function renderFiles() {
    if (filteredFiles.length === 0) {
        filesList.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }
    
    filesList.style.display = 'block';
    emptyState.style.display = 'none';
    
    filesList.innerHTML = '';
    
    filteredFiles.forEach(file => {
        const fileElement = createFileElement(file);
        filesList.appendChild(fileElement);
    });
}

function createFileElement(file) {
    const div = document.createElement('div');
    div.className = 'file-item';
    div.dataset.fileId = file.name;
    
    const fileIcon = getFileIcon(file.type);
    const statusClass = getStatusClass(file);
    const statusText = getStatusText(file);
    
    div.innerHTML = `
        <div class="file-icon ${file.type}">
            <i class="${fileIcon}"></i>
        </div>
        <div class="file-info">
            <div class="file-name" title="${file.name}">${file.name}</div>
            <div class="file-meta">
                <span>${formatFileSize(file.size)}</span>
                <span>${formatDate(file.uploadDate)}</span>
            </div>
        </div>
        <div class="file-status ${statusClass}">
            <i class="fas fa-circle"></i>
            ${statusText}
        </div>
        <div class="file-actions">
            ${(file.hasEmbeddings || file.hasTranscript) ? 
                `<button class="file-action load-doc" title="Load into Chat" onclick="loadDocumentIntoChat('${file.name}')">
                    <i class="fas fa-comments"></i>
                </button>` : ''
            }
            <button class="file-action" title="View Details" onclick="showFileDetails('${file.name}')">
                <i class="fas fa-info-circle"></i>
            </button>
            <button class="file-action" title="Download" onclick="downloadFile('${file.name}')">
                <i class="fas fa-download"></i>
            </button>
            <button class="file-action danger" title="Delete" onclick="deleteFile('${file.name}')">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;
    
    // Add click handler for selection
    div.addEventListener('click', (e) => {
        if (!e.target.closest('.file-action')) {
            toggleFileSelection(file.name, div);
        }
    });
    
    return div;
}

function getFileIcon(type) {
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

function getStatusClass(file) {
    if (file.hasEmbeddings) return 'ready';
    if (file.hasTranscript) return 'ready';
    return 'processing';
}

function getStatusText(file) {
    if (file.hasEmbeddings || file.hasTranscript) return 'Ready';
    return 'Processing';
}

function formatFileSize(size) {
    if (!size) return 'Unknown size';
    
    const sizeInMB = parseFloat(size.replace('MB', ''));
    if (sizeInMB < 1) {
        return `${Math.round(sizeInMB * 1024)}KB`;
    }
    return `${sizeInMB.toFixed(1)}MB`;
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown date';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return 'Today';
    if (diffDays === 2) return 'Yesterday';
    if (diffDays <= 7) return `${diffDays} days ago`;
    
    return date.toLocaleDateString();
}

function toggleFileSelection(fileName, element) {
    if (selectedFiles.has(fileName)) {
        selectedFiles.delete(fileName);
        element.classList.remove('selected');
    } else {
        selectedFiles.add(fileName);
        element.classList.add('selected');
    }
}

function updateStatistics() {
    const pdfs = allFiles.filter(file => file.type === 'pdf').length;
    const audio = allFiles.filter(file => file.type === 'audio').length;
    const video = allFiles.filter(file => file.type === 'video').length;
    const processed = allFiles.filter(file => 
        file.hasEmbeddings || file.hasTranscript
    ).length;
    
    pdfCount.textContent = pdfs;
    audioCount.textContent = audio;
    videoCount.textContent = video;
    processedCount.textContent = processed;
}

function updateFileCount() {
    const count = filteredFiles.length;
    filesCount.textContent = `${count} file${count !== 1 ? 's' : ''}`;
}

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
    batchInput.value = '';
    uploadProgress.style.display = 'none';
    progressFill.style.width = '0%';
    progressPercentage.textContent = '0%';
    progressDetails.innerHTML = '';
    isUploading = false;
}

function setupDragAndDrop() {
    [pdfUploadArea, mediaUploadArea, batchUploadArea].forEach(area => {
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
            } else if (area === mediaUploadArea) {
                handleMediaFiles(files);
            } else {
                handleBatchFiles(files);
            }
        });
    });
}

function handlePdfUpload(e) {
    const files = Array.from(e.target.files);
    handlePdfFiles(files);
}

function handleMediaUpload(e) {
    const files = Array.from(e.target.files);
    handleMediaFiles(files);
}

function handleBatchUpload(e) {
    const files = Array.from(e.target.files);
    handleBatchFiles(files);
}

async function handlePdfFiles(files) {
    const pdfFiles = files.filter(file => file.type === 'application/pdf');
    if (pdfFiles.length === 0) {
        showNotification('Please select PDF files only.', 'error');
        return;
    }
    
    await uploadFiles(pdfFiles, 'pdf');
}

async function handleMediaFiles(files) {
    const mediaFiles = files.filter(file => 
        file.type.startsWith('audio/') || file.type.startsWith('video/')
    );
    if (mediaFiles.length === 0) {
        showNotification('Please select audio or video files only.', 'error');
        return;
    }
    
    await uploadFiles(mediaFiles, 'media');
}

async function handleBatchFiles(files) {
    const validFiles = files.filter(file => 
        file.type === 'application/pdf' || 
        file.type.startsWith('audio/') || 
        file.type.startsWith('video/')
    );
    
    if (validFiles.length === 0) {
        showNotification('Please select valid files (PDF, audio, or video).', 'error');
        return;
    }
    
    await uploadFiles(validFiles, 'batch');
}

async function uploadFiles(files, type) {
    if (isUploading) return;
    
    isUploading = true;
    uploadProgress.style.display = 'block';
    
    let successCount = 0;
    let errorCount = 0;
    
    try {
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const progress = ((i + 1) / files.length) * 100;
            
            progressFill.style.width = `${progress}%`;
            progressPercentage.textContent = `${Math.round(progress)}%`;
            progressDetails.innerHTML = `
                <div>Uploading: ${file.name}</div>
                <div>Progress: ${i + 1} of ${files.length} files</div>
                <div>Success: ${successCount}, Errors: ${errorCount}</div>
            `;
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                let endpoint;
                if (type === 'pdf' || (type === 'batch' && file.type === 'application/pdf')) {
                    endpoint = '/upload_pdf';
                } else {
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
                
                successCount++;
                
            } catch (error) {
                console.error(`Error uploading ${file.name}:`, error);
                errorCount++;
            }
            
            // Small delay between uploads
            await new Promise(resolve => setTimeout(resolve, 200));
        }
        
        progressDetails.innerHTML = `
            <div>Upload completed!</div>
            <div>Success: ${successCount} files</div>
            <div>Errors: ${errorCount} files</div>
        `;
        
        if (successCount > 0) {
            showNotification(`Successfully uploaded ${successCount} file(s)`, 'success');
            addLogEntry(`📤 Uploaded ${successCount} file(s) successfully`, 'success');
            setTimeout(() => {
                hideUploadModal();
                loadAllFiles();
                // Start progress tracking for uploaded files
                startProgressTracking();
            }, 1500);
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('Upload failed', 'error');
    } finally {
        isUploading = false;
    }
}

function showFileDetails(fileName) {
    const file = allFiles.find(f => f.name === fileName);
    if (!file) return;
    
    const modalContent = document.getElementById('fileDetailsContent');
    modalContent.innerHTML = `
        <div class="file-details">
            <div class="detail-section">
                <h4>File Information</h4>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>Name:</label>
                        <span>${file.name}</span>
                    </div>
                    <div class="detail-item">
                        <label>Type:</label>
                        <span>${file.type.toUpperCase()}</span>
                    </div>
                    <div class="detail-item">
                        <label>Size:</label>
                        <span>${file.size}</span>
                    </div>
                    <div class="detail-item">
                        <label>Status:</label>
                        <span class="file-status ${getStatusClass(file)}">
                            ${getStatusText(file)}
                        </span>
                    </div>
                </div>
            </div>
            
            ${file.hasEmbeddings || file.hasTranscript ? `
                <div class="detail-section">
                    <h4>Processing Status</h4>
                    <div class="status-info">
                        <div class="status-item">
                            <i class="fas fa-check-circle"></i>
                            <span>Successfully processed</span>
                        </div>
                        <div class="status-item">
                            <i class="fas fa-search"></i>
                            <span>Ready for chat queries</span>
                        </div>
                    </div>
                </div>
            ` : `
                <div class="detail-section">
                    <h4>Processing Status</h4>
                    <div class="status-info">
                        <div class="status-item processing">
                            <i class="fas fa-spinner fa-spin"></i>
                            <span>Currently processing...</span>
                        </div>
                        <div class="status-note">
                            This file is being processed in the background. 
                            You can check back later or refresh the page.
                        </div>
                    </div>
                </div>
            `}
            
            <div class="detail-actions">
                ${(file.hasEmbeddings || file.hasTranscript) ? `
                    <button class="action-btn load-chat" onclick="loadDocumentIntoChat('${fileName}')">
                        <i class="fas fa-comments"></i>
                        Load into Chat
                    </button>
                ` : ''}
                <button class="action-btn primary" onclick="downloadFile('${fileName}')">
                    <i class="fas fa-download"></i>
                    Download
                </button>
                <button class="action-btn" onclick="deleteFile('${fileName}')">
                    <i class="fas fa-trash"></i>
                    Delete
                </button>
            </div>
        </div>
    `;
    
    fileDetailsModal.classList.add('show');
}

function hideFileDetailsModal() {
    fileDetailsModal.classList.remove('show');
}

async function downloadFile(fileName) {
    try {
        const response = await fetch(`/download_document?filename=${encodeURIComponent(fileName)}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            showNotification(`Downloaded ${fileName}`, 'success');
        } else {
            const data = await response.json();
            throw new Error(data.error || 'Download failed');
        }
    } catch (error) {
        console.error('Download error:', error);
        showNotification(`Download failed: ${error.message}`, 'error');
    }
}

async function deleteFile(fileName) {
    if (confirm(`Are you sure you want to delete "${fileName}"?`)) {
        try {
            const response = await fetch('/delete_document', {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: fileName })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showNotification(data.message, 'success');
                // Refresh the file list
                loadAllFiles();
            } else {
                throw new Error(data.error || 'Delete failed');
            }
        } catch (error) {
            console.error('Delete error:', error);
            showNotification(`Delete failed: ${error.message}`, 'error');
        }
    }
}

function toggleView() {
    // This would implement view toggle functionality (list vs grid)
    showNotification('View toggle not implemented yet', 'info');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    // Remove notification
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
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
    
    /* Load Document Button Styling */
    .file-action.load-doc {
        background-color: #10a37f !important;
        color: white !important;
        border-color: #10a37f !important;
    }
    
    .file-action.load-doc:hover {
        background-color: #0d8f6f !important;
        border-color: #0d8f6f !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(16, 163, 127, 0.3);
    }
    
    .file-action.load-doc i {
        color: white !important;
    }
    
    /* Load Chat Button in Modal */
    .action-btn.load-chat {
        background-color: #10a37f !important;
        color: white !important;
        border-color: #10a37f !important;
        order: -1; /* Put it first */
    }
    
    .action-btn.load-chat:hover {
        background-color: #0d8f6f !important;
        border-color: #0d8f6f !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(16, 163, 127, 0.3);
    }
    
    .action-btn.load-chat i {
        color: white !important;
    }
`;

// Add styles to page
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Load document into chat functionality
async function loadDocumentIntoChat(fileName) {
    try {
        showNotification('Loading document into chat...', 'info');
        
        // Remove file extension to get the title
        const title = fileName.replace(/\.(pdf|mp3|mp4|wav|m4a|avi|mov)$/i, '');
        
        const response = await fetch('/load_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: title })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                showNotification(`✅ Document "${title}" loaded into chat successfully!`, 'success');
                
                // Store the loaded document name in localStorage for persistence
                localStorage.setItem('loadedDocument', result.document_title || title);
                
                // Switch to chat tab after a short delay
                setTimeout(() => {
                    // Try to switch to chat tab if it exists
                    const chatTab = document.querySelector('a[href="/"]');
                    if (chatTab) {
                        chatTab.click();
                    } else {
                        // If no chat tab found, redirect to chat page
                        window.location.href = '/chat.html';
                    }
                }, 1500);
            } else {
                showNotification(`❌ Failed to load document: ${result.error || 'Unknown error'}`, 'error');
            }
        } else {
            const error = await response.json();
            showNotification(`❌ Error loading document: ${error.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Error loading document:', error);
        showNotification(`❌ Error loading document: ${error.message}`, 'error');
    }
}

// Progress tracking functions
function startProgressTracking() {
    // Check for any files that are currently processing
    fetch('/get_existing_documents')
        .then(response => response.json())
        .then(data => {
            if (data.documents) {
                let processingCount = 0;
                data.documents.forEach(doc => {
                    // Check if file has embeddings file but shows as processing
                    const hasEmbeddingsFile = checkEmbeddingsFile(doc.name);
                    if (!doc.hasEmbeddings && !doc.hasTranscript && !hasEmbeddingsFile) {
                        // File is processing, start tracking progress
                        checkDocumentProgress(doc.name);
                        processingCount++;
                    } else if (!doc.hasEmbeddings && !doc.hasTranscript && hasEmbeddingsFile) {
                        // File completed but UI not updated, refresh it
                        addLogEntry(`✅ File ${doc.name} completed, refreshing status`, 'success');
                        setTimeout(() => loadAllFiles(), 1000);
                    }
                });
                if (processingCount > 0) {
                    addLogEntry(`🔄 Started progress tracking for ${processingCount} file(s)`, 'info');
                }
            }
        })
        .catch(error => {
            console.error('Error starting progress tracking:', error);
            addLogEntry(`❌ Error starting progress tracking: ${error.message}`, 'error');
        });
}

function checkEmbeddingsFile(filename) {
    // Check if embeddings file exists for this document
    const title = filename.replace(/\.pdf$/i, '');
    return fetch(`/embedding_progress/${encodeURIComponent(title)}`)
        .then(response => response.json())
        .then(data => {
            // If we get a response and status is completed, file exists
            return data.status === 'completed' || (data.status !== 'not_found' && data.progress >= 100);
        })
        .catch(() => false);
}

// Force refresh for completed files
function refreshCompletedFiles() {
    addLogEntry(`🔄 Checking for completed files...`, 'info');
    setTimeout(() => {
        loadAllFiles();
        addLogEntry(`📊 Refreshed file list`, 'info');
    }, 1000);
}


function updateFileProgress(title, progress, status) {
    // Find the file element and update its progress
    const fileElements = document.querySelectorAll('.file-item');
    fileElements.forEach(element => {
        if (element.dataset.fileId === title) {
            const statusElement = element.querySelector('.file-status');
            if (statusElement) {
                statusElement.innerHTML = `
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${Math.min(100, Math.max(0, progress))}%"></div>
                        </div>
                        <div class="progress-text">${status} (${Math.round(progress)}%)</div>
                    </div>
                `;
            }
        }
    });
}

function updateFileStatus(title, status, statusText) {
    // Find the file element and update its status
    const fileElements = document.querySelectorAll('.file-item');
    fileElements.forEach(element => {
        if (element.dataset.fileId === title) {
            const statusElement = element.querySelector('.file-status');
            if (statusElement) {
                statusElement.className = `file-status ${status}`;
                statusElement.innerHTML = `<i class="fas fa-circle"></i>${statusText}`;
            }
        }
    });
}

// Log management functions
function addLogEntry(message, type = 'info') {
    const logsContent = document.getElementById('logsContent');
    if (!logsContent) return;
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    logEntry.innerHTML = `
        <span class="log-time">${timeString}</span>
        <span class="log-message">${message}</span>
    `;
    
    logsContent.appendChild(logEntry);
    
    // Auto-scroll to bottom
    logsContent.scrollTop = logsContent.scrollHeight;
    
    // Keep only last 50 entries
    const entries = logsContent.querySelectorAll('.log-entry');
    if (entries.length > 50) {
        entries[0].remove();
    }
}

function clearLogs() {
    const logsContent = document.getElementById('logsContent');
    if (logsContent) {
        logsContent.innerHTML = `
            <div class="log-entry">
                <span class="log-time">System ready</span>
                <span class="log-message">Logs cleared</span>
            </div>
        `;
    }
}

// Enhanced progress tracking with logs
function checkDocumentProgress(title) {
    // Remove .pdf extension to match backend title format
    const backendTitle = title.replace(/\.pdf$/i, '');
    
    fetch(`/embedding_progress/${encodeURIComponent(backendTitle)}`)
        .then(response => response.json())
        .then(data => {
            console.log(`Progress for ${backendTitle}:`, data);
            addLogEntry(`Processing ${title}: ${data.status} (${Math.round(data.progress || 0)}%)`, 'info');
            
            if (data.status === 'completed') {
                // Update UI to show completion
                updateFileStatus(title, 'ready', 'Ready');
                addLogEntry(`✅ Successfully processed ${title}`, 'success');
                // Refresh the files list
                setTimeout(() => loadAllFiles(), 1000);
            } else if (data.status === 'error') {
                console.error(`Error processing ${title}:`, data.error);
                updateFileStatus(title, 'error', `Error: ${data.error}`);
                addLogEntry(`❌ Error processing ${title}: ${data.error}`, 'error');
            } else if (data.status !== 'not_found' && data.status !== 'completed') {
                // Update progress display
                updateFileProgress(title, data.progress || 0, data.status || 'Processing...');
                // Continue checking progress
                setTimeout(() => checkDocumentProgress(title), 2000);
            }
        })
        .catch(error => {
            console.error('Error checking progress:', error);
            addLogEntry(`❌ Error checking progress for ${title}: ${error.message}`, 'error');
        });
}
