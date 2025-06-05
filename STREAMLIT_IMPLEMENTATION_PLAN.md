# 🚀 Streamlit-Optimized Multi-Document Implementation Plan

## 📋 **Executive Summary**

This plan transforms the FinDocPipeline into a Streamlit-native multi-document processor that handles up to 20 documents concurrently (reduced from 50 for Streamlit compatibility) with automatic document type classification and enhanced schema design.

**Key Constraints Addressed:**
- Streamlit's threading limitations (max 4 concurrent workers)
- Memory constraints (< 1GB total usage)
- Pure Python dependencies only
- Session state management for progress tracking

---

## 🎯 **Phase 1: Foundation & Multi-File Support**
*Duration: 1-2 weeks*

### **1.1 Enhanced File Upload Interface**
```python
# File: ui/multi_upload_streamlit.py
import streamlit as st
from typing import List, Dict, Any
import time

class StreamlitMultiUpload:
    """Streamlit-native multi-file upload with preview"""
    
    def render(self) -> List[st.runtime.uploaded_file_manager.UploadedFile]:
        """Render enhanced upload interface"""
        
        st.header("📁 Upload Financial Documents")
        st.markdown("*Support for PDF, DOCX, TXT, CSV, XLSX - Up to 20 documents*")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
            accept_multiple_files=True,
            help="Upload earnings transcripts, presentations, financial summaries"
        )
        
        if uploaded_files:
            # Validate file count
            if len(uploaded_files) > 20:
                st.error("⚠️ Maximum 20 files allowed. Please remove some files.")
                return uploaded_files[:20]
            
            # Display file queue
            self._render_file_queue(uploaded_files)
            
            # Processing configuration
            self._render_processing_config()
        
        return uploaded_files
    
    def _render_file_queue(self, files: List) -> None:
        """Display uploaded files with management options"""
        
        st.subheader(f"📋 Document Queue ({len(files)} files)")
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Quick Classify"):
                self._quick_classify_all(files)
        with col2:
            if st.button("📊 Preview All"):
                self._preview_all_files(files)
        with col3:
            if st.button("🗑️ Clear Queue"):
                st.rerun()
        
        # File list with details
        for i, file in enumerate(files):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.write(f"📄 **{file.name}**")
                    file_size_mb = file.size / (1024 * 1024)
                    st.caption(f"Size: {file_size_mb:.2f} MB")
                
                with col2:
                    # Quick type detection
                    doc_type = self._detect_type_from_name(file.name)
                    type_colors = {
                        'transcript': '🎙️',
                        'presentation': '📊', 
                        'financial_summary': '📈',
                        'data': '📋',
                        'unknown': '❓'
                    }
                    st.write(f"{type_colors.get(doc_type, '❓')} {doc_type.title()}")
                
                with col3:
                    # Priority
                    priority = st.selectbox(
                        "Priority",
                        ["High", "Normal", "Low"],
                        index=1,
                        key=f"priority_{i}",
                        label_visibility="collapsed"
                    )
                
                with col4:
                    # Preview
                    if st.button("👁️", key=f"preview_{i}"):
                        self._show_file_preview(file)
                
                st.divider()
    
    def _detect_type_from_name(self, filename: str) -> str:
        """Quick type detection from filename"""
        name_lower = filename.lower()
        
        if any(word in name_lower for word in ['transcript', 'call', 'earnings']):
            return 'transcript'
        elif any(word in name_lower for word in ['presentation', 'slides', 'deck']):
            return 'presentation'
        elif any(word in name_lower for word in ['summary', 'financial', 'statement']):
            return 'financial_summary'
        elif filename.endswith(('.csv', '.xlsx')):
            return 'data'
        else:
            return 'unknown'
```

### **1.2 Streamlit-Safe File Handlers**
```python
# File: file_handlers/streamlit_handlers.py
import streamlit as st
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pdfplumber
import pandas as pd
from docx import Document
import io

class StreamlitFileHandler(ABC):
    """Base handler optimized for Streamlit"""
    
    @abstractmethod
    def extract_content(self, uploaded_file) -> Dict[str, Any]:
        """Extract content from Streamlit uploaded file"""
        pass
    
    @st.cache_data
    def get_file_info(_self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Cached file information extraction"""
        return {
            'size': len(file_bytes),
            'filename': filename,
            'type': filename.split('.')[-1].lower()
        }

class StreamlitPDFHandler(StreamlitFileHandler):
    """PDF processing optimized for Streamlit"""
    
    def extract_content(self, uploaded_file) -> Dict[str, Any]:
        """Extract PDF content with caching"""
        
        # Read file bytes
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for potential reuse
        
        # Use cached extraction
        return self._extract_pdf_cached(file_bytes, uploaded_file.name)
    
    @st.cache_data
    def _extract_pdf_cached(_self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Cached PDF extraction"""
        
        try:
            # Create file-like object
            pdf_file = io.BytesIO(file_bytes)
            
            # Extract with pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                text_content = []
                tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            tables.append({
                                'page': page_num + 1,
                                'data': table
                            })
                
                full_text = '\n'.join(text_content)
                
                return {
                    'text': full_text,
                    'tables': tables,
                    'page_count': len(pdf.pages),
                    'metadata': {
                        'filename': filename,
                        'pages': len(pdf.pages),
                        'has_tables': len(tables) > 0
                    }
                }
        
        except Exception as e:
            return {
                'text': '',
                'tables': [],
                'page_count': 0,
                'metadata': {'filename': filename, 'error': str(e)}
            }

class StreamlitDOCXHandler(StreamlitFileHandler):
    """DOCX processing optimized for Streamlit"""
    
    def extract_content(self, uploaded_file) -> Dict[str, Any]:
        """Extract DOCX content with caching"""
        
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        return self._extract_docx_cached(file_bytes, uploaded_file.name)
    
    @st.cache_data
    def _extract_docx_cached(_self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Cached DOCX extraction"""
        
        try:
            # Create file-like object
            docx_file = io.BytesIO(file_bytes)
            
            # Extract with python-docx
            doc = Document(docx_file)
            
            # Extract paragraphs
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = '\n'.join(paragraphs)
            
            # Extract tables
            tables = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
                tables.append({'data': table_data})
            
            return {
                'text': full_text,
                'tables': tables,
                'paragraph_count': len(paragraphs),
                'metadata': {
                    'filename': filename,
                    'paragraphs': len(paragraphs),
                    'tables': len(tables)
                }
            }
        
        except Exception as e:
            return {
                'text': '',
                'tables': [],
                'paragraph_count': 0,
                'metadata': {'filename': filename, 'error': str(e)}
            }

class StreamlitExcelHandler(StreamlitFileHandler):
    """Excel processing optimized for Streamlit"""
    
    def extract_content(self, uploaded_file) -> Dict[str, Any]:
        """Extract Excel content with caching"""
        
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        return self._extract_excel_cached(file_bytes, uploaded_file.name)
    
    @st.cache_data
    def _extract_excel_cached(_self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Cached Excel extraction"""
        
        try:
            # Create file-like object
            excel_file = io.BytesIO(file_bytes)
            
            # Read all sheets
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            
            # Process each sheet
            sheets_text = []
            tables = []
            
            for sheet_name, df in excel_data.items():
                # Convert to text
                sheet_text = f"Sheet: {sheet_name}\n"
                sheet_text += df.to_string(index=False)
                sheets_text.append(sheet_text)
                
                # Store as table
                tables.append({
                    'sheet': sheet_name,
                    'data': df.values.tolist(),
                    'columns': df.columns.tolist()
                })
            
            full_text = '\n\n'.join(sheets_text)
            
            return {
                'text': full_text,
                'tables': tables,
                'sheet_count': len(excel_data),
                'metadata': {
                    'filename': filename,
                    'sheets': list(excel_data.keys()),
                    'total_rows': sum(len(df) for df in excel_data.values())
                }
            }
        
        except Exception as e:
            return {
                'text': '',
                'tables': [],
                'sheet_count': 0,
                'metadata': {'filename': filename, 'error': str(e)}
            }
```

---

## 🎯 **Phase 2: Document Classification System**
*Duration: 1 week*

### **2.1 Lightweight Classification Engine**
```python
# File: classification/streamlit_classifier.py
import streamlit as st
from typing import Dict, Any, List
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class StreamlitDocumentClassifier:
    """Lightweight document classifier for Streamlit"""
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        
        model_path = "models/document_classifier.pkl"
        
        if os.path.exists(model_path):
            self._load_models(model_path)
        else:
            self._create_and_train_models()
    
    @st.cache_resource
    def _create_and_train_models(_self):
        """Create and train classification models"""
        
        # Training data (embedded in code for simplicity)
        training_data = _self._get_training_data()
        
        # Create vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Create classifier
        classifier = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=2
        )
        
        # Train
        texts = [item['text'] for item in training_data]
        labels = [item['label'] for item in training_data]
        
        X = vectorizer.fit_transform(texts)
        classifier.fit(X, labels)
        
        return vectorizer, classifier
    
    def classify_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document with multiple approaches"""
        
        text = content.get('text', '')
        metadata = content.get('metadata', {})
        
        # Rule-based classification (fast)
        rule_result = self._rule_based_classification(text, metadata)
        
        # ML classification (if rule-based is uncertain)
        if rule_result['confidence'] < 0.8:
            ml_result = self._ml_classification(text)
            
            # Combine results
            if ml_result['confidence'] > rule_result['confidence']:
                return ml_result
        
        return rule_result
    
    def _rule_based_classification(self, text: str, metadata: Dict) -> Dict[str, Any]:
        """Fast rule-based classification"""
        
        text_lower = text.lower()
        filename = metadata.get('filename', '').lower()
        
        # Scoring system
        scores = {
            'transcript': 0,
            'presentation': 0,
            'financial_summary': 0
        }
        
        # Transcript indicators
        transcript_patterns = [
            r'operator[:\s]',
            r'analyst[:\s]',
            r'q&a|question.{0,10}answer',
            r'earnings call|conference call',
            r'thank you.{0,20}operator',
            r'next question',
            r'[A-Z][a-z]+ [A-Z][a-z]+:', # Speaker patterns
        ]
        
        for pattern in transcript_patterns:
            matches = len(re.findall(pattern, text_lower))
            scores['transcript'] += matches * 2
        
        # Presentation indicators
        presentation_patterns = [
            r'slide \d+|next slide',
            r'agenda|overview',
            r'key highlights|financial highlights',
            r'moving to slide|turn to slide',
            r'as you can see on the slide'
        ]
        
        for pattern in presentation_patterns:
            matches = len(re.findall(pattern, text_lower))
            scores['presentation'] += matches * 2
        
        # Financial summary indicators
        summary_patterns = [
            r'income statement|profit.{0,10}loss',
            r'balance sheet',
            r'cash flow statement',
            r'financial statements?',
            r'quarterly results?',
            r'revenue.*\$|net income.*\$'
        ]
        
        for pattern in summary_patterns:
            matches = len(re.findall(pattern, text_lower))
            scores['financial_summary'] += matches * 2
        
        # Filename indicators
        if any(word in filename for word in ['transcript', 'call', 'earnings']):
            scores['transcript'] += 5
        elif any(word in filename for word in ['presentation', 'slides', 'deck']):
            scores['presentation'] += 5
        elif any(word in filename for word in ['summary', 'financial', 'statement']):
            scores['financial_summary'] += 5
        
        # Determine result
        max_score = max(scores.values())
        if max_score == 0:
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'method': 'rule_based',
                'scores': scores
            }
        
        predicted_type = max(scores, key=scores.get)
        confidence = min(max_score / 10.0, 1.0)  # Normalize to 0-1
        
        return {
            'document_type': predicted_type,
            'confidence': confidence,
            'method': 'rule_based',
            'scores': scores
        }
    
    @st.cache_data
    def _ml_classification(_self, text: str) -> Dict[str, Any]:
        """ML-based classification with caching"""
        
        if not _self.vectorizer or not _self.classifier:
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'method': 'ml_unavailable'
            }
        
        # Vectorize text
        X = _self.vectorizer.transform([text])
        
        # Predict
        prediction = _self.classifier.predict(X)[0]
        probabilities = _self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return {
            'document_type': prediction,
            'confidence': confidence,
            'method': 'machine_learning',
            'probabilities': dict(zip(_self.classifier.classes_, probabilities))
        }
```

---

## 🎯 **Phase 3: Concurrent Processing Engine**
*Duration: 1 week*

### **3.1 Streamlit-Safe Concurrent Manager**
```python
# File: processing/streamlit_concurrent.py
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import List, Dict, Any, Callable
import queue

class StreamlitConcurrentProcessor:
    """Streamlit-safe concurrent document processor"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = min(max_workers, 4)  # Streamlit limit
        self.batch_size = 5  # Process in small batches
        self.progress_queue = queue.Queue()
        
    def process_documents_batch(
        self, 
        uploaded_files: List,
        processor_func: Callable,
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """Process documents in Streamlit-safe batches"""
        
        # Initialize progress tracking
        total_files = len(uploaded_files)
        results = {}
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_container = st.empty()
        results_container = st.empty()
        
        # Process in batches
        for batch_start in range(0, total_files, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_files)
            batch = uploaded_files[batch_start:batch_end]
            
            # Update status
            batch_num = (batch_start // self.batch_size) + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            
            status_container.text(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} documents)..."
            )
            
            # Process batch concurrently
            batch_results = self._process_batch_concurrent(
                batch, processor_func, progress_callback
            )
            
            # Update results
            results.update(batch_results)
            
            # Update progress
            progress = batch_end / total_files
            progress_bar.progress(progress)
            
            # Update results display
            self._update_results_display(results_container, results)
            
            # Small delay to prevent overwhelming
            time.sleep(0.1)
        
        # Final status update
        status_container.text(f"✅ Completed processing {total_files} documents!")
        progress_bar.progress(1.0)
        
        return results
    
    def _process_batch_concurrent(
        self, 
        batch: List,
        processor_func: Callable,
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """Process a batch of files concurrently"""
        
        results = {}
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Submit all tasks in the batch
            future_to_file = {}
            for file in batch:
                future = executor.submit(self._process_single_file, file, processor_func)
                future_to_file[future] = file
            
            # Collect results as they complete
            for future in as_completed(future_to_file, timeout=120):  # 2 minute timeout
                file = future_to_file[future]
                
                try:
                    result = future.result()
                    results[file.name] = result
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(file.name, result, None)
                
                except Exception as e:
                    error_result = {
                        'error': str(e),
                        'status': 'failed',
                        'filename': file.name
                    }
                    results[file.name] = error_result
                    
                    # Call progress callback for error
                    if progress_callback:
                        progress_callback(file.name, None, str(e))
        
        return results
    
    def _process_single_file(
        self, 
        uploaded_file,
        processor_func: Callable
    ) -> Dict[str, Any]:
        """Process a single file with error handling"""
        
        start_time = time.time()
        
        try:
            # Process the file
            result = processor_func(uploaded_file)
            
            # Add processing metadata
            result['processing_time'] = time.time() - start_time
            result['status'] = 'completed'
            result['filename'] = uploaded_file.name
            
            return result
        
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed',
                'filename': uploaded_file.name,
                'processing_time': time.time() - start_time
            }
    
    def _update_results_display(self, container, results: Dict) -> None:
        """Update the results display in real-time"""
        
        if not results:
            return
        
        # Count status
        completed = sum(1 for r in results.values() if r.get('status') == 'completed')
        failed = sum(1 for r in results.values() if r.get('status') == 'failed')
        total = len(results)
        
        # Display summary
        with container.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Completed", completed, f"{completed/total:.1%}" if total > 0 else "0%")
            
            with col2:
                st.metric("Failed", failed, f"{failed/total:.1%}" if total > 0 else "0%")
            
            with col3:
                st.metric("Total", total)
            
            # Show recent results
            if results:
                st.subheader("Recent Results")
                recent_results = list(results.items())[-3:]  # Show last 3
                
                for filename, result in recent_results:
                    status = result.get('status', 'unknown')
                    if status == 'completed':
                        st.success(f"✅ {filename}")
                    elif status == 'failed':
                        st.error(f"❌ {filename}: {result.get('error', 'Unknown error')}")
```

This implementation plan provides a Streamlit-native approach that works within the platform's constraints while delivering robust multi-document processing capabilities. The key is using lightweight dependencies, limited concurrency, and proper session state management.

Would you like me to continue with Phase 4 (Enhanced Schema Design) or would you prefer to focus on implementing one of these phases first?