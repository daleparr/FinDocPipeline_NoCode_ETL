# 🏗️ FinDocPipeline Multi-Document Project Structure

## 📁 **Enhanced Project Directory Structure**

```
FinDocPipeline/
├── README.md                           # Updated project documentation
├── requirements.txt                    # Current dependencies
├── requirements_enhanced.txt           # New enhanced dependencies
├── packages.txt                        # System packages
├── .streamlit/
│   └── config.toml                    # Streamlit configuration
├── .gitignore                         # Git ignore patterns
│
├── FinDocPipeline.py                  # Current main application
├── FinDocPipeline_Enhanced.py         # New multi-document version
│
├── file_handlers/                     # Multi-format file processing
│   ├── __init__.py
│   ├── base_handler.py               # Abstract base class
│   ├── pdf_handler.py                # Enhanced PDF processing
│   ├── docx_handler.py               # Microsoft Word processing
│   ├── txt_handler.py                # Plain text processing
│   ├── csv_handler.py                # CSV data processing
│   ├── excel_handler.py              # Excel file processing
│   └── file_detector.py              # File type detection
│
├── classification/                    # Document type classification
│   ├── __init__.py
│   ├── document_classifier.py        # Main classification logic
│   ├── feature_extractors.py         # Content feature extraction
│   ├── models/                       # Pre-trained models
│   │   ├── transcript_model.pkl
│   │   ├── presentation_model.pkl
│   │   └── summary_model.pkl
│   └── training_data/                # Classification training data
│       ├── transcript_samples/
│       ├── presentation_samples/
│       └── summary_samples/
│
├── schemas/                          # Data schemas and validation
│   ├── __init__.py
│   ├── base_schema.py               # Base schema definitions
│   ├── transcript_schema.py         # Transcript-specific schema
│   ├── presentation_schema.py       # Presentation-specific schema
│   ├── summary_schema.py            # Financial summary schema
│   └── validators.py                # Schema validation logic
│
├── processing/                       # Concurrent processing engine
│   ├── __init__.py
│   ├── concurrent_manager.py        # Main concurrent processing
│   ├── task_queue.py               # Task queue management
│   ├── progress_tracker.py         # Progress tracking
│   ├── resource_manager.py         # Resource allocation
│   ├── result_aggregator.py        # Results consolidation
│   └── type_processors/            # Type-specific processors
│       ├── __init__.py
│       ├── transcript_processor.py  # Transcript processing
│       ├── presentation_processor.py # Presentation processing
│       └── summary_processor.py     # Financial summary processing
│
├── ui/                              # Enhanced user interface
│   ├── __init__.py
│   ├── multi_upload.py             # Multi-file upload interface
│   ├── progress_dashboard.py       # Real-time progress tracking
│   ├── results_viewer.py           # Results visualization
│   └── batch_export.py             # Batch export functionality
│
├── analytics/                       # Cross-document analytics
│   ├── __init__.py
│   ├── cross_document_analyzer.py  # Multi-document analysis
│   ├── trend_analyzer.py           # Financial trend analysis
│   ├── topic_analyzer.py           # Topic modeling across docs
│   └── comparative_metrics.py      # Comparative analysis
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── text_processing.py          # Text processing utilities
│   ├── financial_extractors.py     # Financial data extraction
│   ├── embedding_utils.py          # Semantic embedding utilities
│   └── export_utils.py             # Export functionality
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_file_handlers.py       # File handler tests
│   ├── test_classification.py      # Classification tests
│   ├── test_processing.py          # Processing tests
│   ├── test_schemas.py             # Schema validation tests
│   └── sample_data/                # Test data files
│       ├── sample_transcript.pdf
│       ├── sample_presentation.pptx
│       ├── sample_summary.docx
│       └── sample_data.xlsx
│
├── docs/                            # Documentation
│   ├── MULTI_DOCUMENT_ARCHITECTURE.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   ├── API_DOCUMENTATION.md
│   ├── SCHEMA_REFERENCE.md
│   └── DEPLOYMENT_GUIDE_ENHANCED.md
│
└── config/                          # Configuration files
    ├── processing_config.yaml       # Processing configuration
    ├── classification_config.yaml   # Classification settings
    └── ui_config.yaml              # UI customization
```

---

## 📋 **Enhanced Requirements Specification**

### **Core Dependencies (requirements_enhanced.txt)**
```python
# Current dependencies
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
pdfplumber>=0.9.0
PyMuPDF>=1.23.0
opencv-python-headless>=4.8.0
pytesseract>=0.3.10
Pillow>=10.0.0
regex>=2023.8.8
openpyxl>=3.1.2

# New multi-format processing
python-docx>=0.8.11              # DOCX processing
xlrd>=2.0.1                      # Excel XLS processing
python-magic>=0.4.27             # File type detection
chardet>=5.2.0                   # Character encoding detection

# Concurrent processing
asyncio-throttle>=1.0.2          # Async processing control
aiofiles>=23.1.0                 # Async file operations
concurrent-futures>=3.1.1        # Enhanced concurrency

# Schema validation and data modeling
pydantic>=2.0.0                  # Schema validation
marshmallow>=3.20.0              # Data serialization
jsonschema>=4.19.0               # JSON schema validation

# Advanced NLP and AI
spacy>=3.7.0                     # Advanced NLP processing
transformers>=4.30.0             # AI-powered classification
sentence-transformers>=2.2.2     # Semantic embeddings
scikit-learn>=1.3.0              # Machine learning
torch>=2.0.0                     # Deep learning framework
huggingface-hub>=0.16.0          # Model hub access

# Financial text processing
nltk>=3.8.1                      # Natural language toolkit
textstat>=0.7.3                  # Text statistics
yfinance>=0.2.18                 # Financial data (optional)

# Data processing and analysis
scipy>=1.11.0                    # Scientific computing
matplotlib>=3.7.0                # Plotting
seaborn>=0.12.0                  # Statistical visualization
plotly>=5.15.0                   # Interactive plots

# Performance and monitoring
memory-profiler>=0.61.0          # Memory usage monitoring
psutil>=5.9.0                    # System resource monitoring
tqdm>=4.65.0                     # Progress bars

# Export and serialization
xlsxwriter>=3.1.0                # Excel export
python-pptx>=0.6.21              # PowerPoint processing
reportlab>=4.0.0                 # PDF generation
```

### **System Packages (packages_enhanced.txt)**
```bash
# Current packages
tesseract-ocr
libgl1-mesa-glx
libglib2.0-0

# New packages for enhanced processing
libmagic1                        # File type detection
libxml2-dev                      # XML processing
libxslt1-dev                     # XSLT processing
libreoffice                      # Office document processing
poppler-utils                    # PDF utilities
```

---

## 🔧 **Technical Implementation Specifications**

### **1. Multi-Format File Handler Architecture**

#### **Base Handler Interface**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"

@dataclass
class ExtractedContent:
    """Standardized content structure"""
    text: str
    metadata: Dict[str, Any]
    structure: Dict[str, Any]
    tables: List[Dict]
    images: List[Dict]
    raw_data: Optional[Any] = None

class BaseFileHandler(ABC):
    """Abstract base for all file handlers"""
    
    @abstractmethod
    def extract_content(self, file_path: str) -> ExtractedContent:
        """Extract standardized content from file"""
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate file can be processed"""
        pass
    
    @abstractmethod
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata and information"""
        pass
```

#### **DOCX Handler Specification**
```python
class DOCXHandler(BaseFileHandler):
    """Microsoft Word document processing"""
    
    def extract_content(self, file_path: str) -> ExtractedContent:
        """Extract comprehensive DOCX content"""
        
        # Core extraction methods
        text = self._extract_full_text(file_path)
        tables = self._extract_tables(file_path)
        images = self._extract_images(file_path)
        structure = self._analyze_document_structure(file_path)
        metadata = self._extract_document_metadata(file_path)
        
        return ExtractedContent(
            text=text,
            metadata=metadata,
            structure=structure,
            tables=tables,
            images=images
        )
    
    def _extract_full_text(self, file_path: str) -> str:
        """Extract all text including headers, footers, footnotes"""
        pass
    
    def _extract_tables(self, file_path: str) -> List[Dict]:
        """Extract table data with formatting"""
        pass
    
    def _analyze_document_structure(self, file_path: str) -> Dict:
        """Analyze document structure and hierarchy"""
        pass
```

#### **Excel Handler Specification**
```python
class ExcelHandler(BaseFileHandler):
    """Excel file processing for financial data"""
    
    def extract_content(self, file_path: str) -> ExtractedContent:
        """Extract Excel content with financial pattern recognition"""
        
        worksheets = self._extract_all_worksheets(file_path)
        charts = self._extract_charts(file_path)
        formulas = self._extract_formulas(file_path)
        financial_patterns = self._detect_financial_patterns(worksheets)
        
        # Convert to standardized format
        text = self._worksheets_to_text(worksheets)
        tables = self._worksheets_to_tables(worksheets)
        
        return ExtractedContent(
            text=text,
            metadata={'financial_patterns': financial_patterns},
            structure={'worksheets': len(worksheets), 'charts': len(charts)},
            tables=tables,
            images=[],
            raw_data={'worksheets': worksheets, 'formulas': formulas}
        )
    
    def _detect_financial_patterns(self, worksheets: List[Dict]) -> Dict:
        """Detect financial statement patterns"""
        patterns = {
            'income_statement': False,
            'balance_sheet': False,
            'cash_flow': False,
            'financial_ratios': False,
            'time_series_data': False
        }
        
        # Pattern detection logic
        for ws in worksheets:
            sheet_name = ws['name'].lower()
            data = ws['data']
            
            # Check for financial statement indicators
            if self._is_income_statement(sheet_name, data):
                patterns['income_statement'] = True
            elif self._is_balance_sheet(sheet_name, data):
                patterns['balance_sheet'] = True
            elif self._is_cash_flow(sheet_name, data):
                patterns['cash_flow'] = True
            elif self._has_financial_ratios(data):
                patterns['financial_ratios'] = True
            
            if self._has_time_series_data(data):
                patterns['time_series_data'] = True
        
        return patterns
```

---

### **2. Advanced Document Classification System**

#### **Feature Extraction Pipeline**
```python
class AdvancedFeatureExtractor:
    """Multi-dimensional feature extraction for document classification"""
    
    def __init__(self):
        self.nlp_model = self._load_nlp_model()
        self.embedding_model = self._load_embedding_model()
        self.financial_lexicon = self._load_financial_lexicon()
    
    def extract_comprehensive_features(self, content: ExtractedContent) -> Dict[str, Any]:
        """Extract all feature types for classification"""
        
        features = {
            'structural': self._extract_structural_features(content),
            'linguistic': self._extract_linguistic_features(content),
            'semantic': self._extract_semantic_features(content),
            'financial': self._extract_financial_features(content),
            'metadata': self._extract_metadata_features(content)
        }
        
        return features
    
    def _extract_structural_features(self, content: ExtractedContent) -> List[float]:
        """Document structure analysis"""
        text = content.text
        structure = content.structure
        
        return [
            # Length metrics
            len(text),
            len(text.split()),
            len(text.split('\n')),
            
            # Structure metrics
            structure.get('paragraph_count', 0),
            structure.get('section_count', 0),
            len(content.tables),
            len(content.images),
            
            # Transcript indicators
            self._count_speaker_patterns(text),
            self._count_qa_indicators(text),
            self._count_timestamp_patterns(text),
            
            # Presentation indicators
            self._count_slide_markers(text),
            self._count_bullet_points(text),
            self._count_chart_references(text),
            
            # Financial document indicators
            self._count_financial_statements(text),
            self._count_numerical_tables(content.tables),
            self._count_financial_ratios(text)
        ]
    
    def _extract_semantic_features(self, content: ExtractedContent) -> List[float]:
        """Semantic embedding features"""
        
        # Generate document embedding
        doc_embedding = self.embedding_model.encode(content.text[:1000])
        
        # Topic-specific embeddings
        topic_embeddings = self._generate_topic_embeddings(content.text)
        
        # Combine embeddings
        combined_features = np.concatenate([
            doc_embedding,
            topic_embeddings
        ])
        
        return combined_features.tolist()
```

#### **Multi-Model Classification Ensemble**
```python
class EnsembleDocumentClassifier:
    """Ensemble classifier combining multiple approaches"""
    
    def __init__(self):
        self.feature_classifier = self._load_feature_classifier()
        self.transformer_classifier = self._load_transformer_classifier()
        self.rule_based_classifier = RuleBasedClassifier()
        
        # Ensemble weights
        self.weights = {
            'feature_based': 0.4,
            'transformer': 0.4,
            'rule_based': 0.2
        }
    
    def classify_document(self, content: ExtractedContent) -> Dict[str, Any]:
        """Comprehensive document classification"""
        
        # Get predictions from each classifier
        feature_pred = self._get_feature_prediction(content)
        transformer_pred = self._get_transformer_prediction(content)
        rule_pred = self._get_rule_based_prediction(content)
        
        # Ensemble prediction
        ensemble_pred = self._combine_predictions(
            feature_pred, transformer_pred, rule_pred
        )
        
        # Determine final classification
        doc_types = ['transcript', 'presentation', 'financial_summary']
        predicted_type = doc_types[np.argmax(ensemble_pred)]
        confidence = np.max(ensemble_pred)
        
        return {
            'document_type': predicted_type,
            'confidence': confidence,
            'type_probabilities': dict(zip(doc_types, ensemble_pred)),
            'individual_predictions': {
                'feature_based': dict(zip(doc_types, feature_pred)),
                'transformer': dict(zip(doc_types, transformer_pred)),
                'rule_based': dict(zip(doc_types, rule_pred))
            }
        }
```

---

### **3. Concurrent Processing Architecture**

#### **Task Management System**
```python
class TaskQueue:
    """Advanced task queue with priority and resource management"""
    
    def __init__(self):
        self.pending_tasks = PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Task statistics
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'average_processing_time': 0.0
        }
    
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit task for processing"""
        self.pending_tasks.put((task.priority, task))
        self.stats['total_submitted'] += 1
        return task.task_id
    
    def get_next_task(self) -> Optional[ProcessingTask]:
        """Get next task for processing"""
        if not self.pending_tasks.empty():
            priority, task = self.pending_tasks.get()
            self.active_tasks[task.task_id] = task
            return task
        return None
    
    def complete_task(self, task_id: str, result: Dict) -> None:
        """Mark task as completed"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.result = result
            task.status = ProcessingStatus.COMPLETED
            task.completed_at = time.time()
            
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            self.stats['total_completed'] += 1
```

#### **Resource Management System**
```python
class ResourceManager:
    """Intelligent resource allocation and monitoring"""
    
    def __init__(self, max_memory_mb: int = 8192):
        self.max_memory_mb = max_memory_mb
        self.current_memory_usage = 0
        self.active_allocations = {}
        
        # Resource monitoring
        self.monitor = ResourceMonitor()
        self.monitor.start()
    
    def can_process_task(self, task: ProcessingTask) -> bool:
        """Check if resources are available for task"""
        estimated_memory = self._estimate_memory_usage(task)
        
        return (
            self.current_memory_usage + estimated_memory <= self.max_memory_mb
            and self.monitor.cpu_usage < 90
            and self.monitor.available_memory > estimated_memory
        )
    
    def allocate_resources(self, task: ProcessingTask) -> None:
        """Allocate resources for task"""
        estimated_memory = self._estimate_memory_usage(task)
        
        self.active_allocations[task.task_id] = {
            'memory_mb': estimated_memory,
            'allocated_at': time.time()
        }
        
        self.current_memory_usage += estimated_memory
    
    def release_resources(self, task: ProcessingTask) -> None:
        """Release resources after task completion"""
        if task.task_id in self.active_allocations:
            allocation = self.active_allocations[task.task_id]
            self.current_memory_usage -= allocation['memory_mb']
            del self.active_allocations[task.task_id]
    
    def _estimate_memory_usage(self, task: ProcessingTask) -> int:
        """Estimate memory usage for task"""
        base_memory = 50  # Base memory in MB
        
        # Scale based on file size
        file_size_mb = task.file_size / (1024 * 1024)
        memory_multiplier = {
            'pdf': 3.0,
            'docx': 2.0,
            'xlsx': 4.0,
            'txt': 1.0,
            'csv': 2.5
        }
        
        multiplier = memory_multiplier.get(task.document_type, 2.0)
        estimated_memory = base_memory + (file_size_mb * multiplier)
        
        return int(estimated_memory)
```

---

### **4. Enhanced User Interface Components**

#### **Multi-File Upload Interface**
```python
class MultiFileUploadInterface:
    """Advanced multi-file upload with preview and management"""
    
    def render(self) -> List[UploadedFile]:
        """Render enhanced upload interface"""
        
        st.header("📁 Upload Financial Documents")
        st.markdown("*Support for PDF, DOCX, TXT, CSV, XLSX files - Up to 50 documents*")
        
        # File uploader with multiple format support
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload earnings transcripts, presentations, financial summaries, and data files"
        )
        
        if uploaded_files:
            # Document queue management
            self._render_document_queue(uploaded_files)
            
            # Processing configuration
            self._render_processing_config()
            
            # Batch processing controls
            self._render_batch_controls(uploaded_files)
        
        return uploaded_files
    
    def _render_document_queue(self, files: List[UploadedFile]) -> None:
        """Render document queue with preview and management"""
        
        st.subheader(f"📋 Document Queue ({len(files)} files)")
        
        # Queue management controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🔍 Auto-Classify All"):
                self._auto_classify_documents(files)
        with col2:
            if st.button("📊 Preview All"):
                self._preview_all_documents(files)
        with col3:
            if st.button("🗑️ Clear Queue"):
                st.session_state.uploaded_files = []
                st.rerun()
        
        # Document list with management
        for i, file in enumerate(files):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
                
                with col1:
                    st.write(f"📄 **{file.name}**")
                    st.caption(f"Size: {file.size / 1024:.1f} KB")
                
                with col2:
                    # Document type classification
                    doc_type = self._classify_document_preview(file)
                    type_color = {
                        'transcript': 'blue',
                        'presentation': 'green', 
                        'financial_summary': 'orange',
                        'unknown': 'gray'
                    }
                    st.markdown(f":{type_color.get(doc_type, 'gray')}[{doc_type.title()}]")
                
                with col3:
                    # Priority setting
                    priority = st.selectbox(
                        "Priority",
                        [1, 2, 3, 4, 5],
                        index=2,
                        key=f"priority_{i}",
                        label_visibility="collapsed"
                    )
                
                with col4:
                    # Preview button
                    if st.button("👁️", key=f"preview_{i}", help="Preview document"):
                        self._show_document_preview(file)
                
                with col5:
                    # Remove button
                    if st.button("🗑️", key=f"remove_{i}", help="Remove from queue"):
                        files.remove(file)
                        st.rerun()
                
                st.divider()
```

#### **Real-Time Progress Dashboard**
```python
class ProgressDashboard:
    """Real-time processing progress visualization"""
    
    def render(self, processing_status: Dict[str, Any]) -> None:
        """Render comprehensive progress dashboard"""
        
        st.header("🚀 Concurrent Processing Dashboard")
        
        # Overall progress
        self._render_overall_progress(processing_status)
        
        # Type-specific progress
        self._render_type_progress(processing_status)
        
        # Active tasks monitoring
        self._render_active_tasks(processing_status)
        
        # Performance metrics
        self._render_performance_metrics(processing_status)
        
        # Error handling and retry
        self._render_error_management(processing_status)
    
    def _render_overall_progress(self, status: Dict) -> None:
        """Render overall processing progress"""
        
        total = status.get('total_documents', 0)
        completed = status.get('completed_documents', 0)
        failed = status.get('failed_documents', 0)
        
        if total > 0:
            progress = completed / total
            st.progress(progress)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Documents", total)
            with col2:
                st.metric("Completed", completed, f"{progress:.1%}")
            with col3:
                st.metric("Failed", failed, f"{failed/total:.1%}" if total > 0 else "0%")
            with col4:
                remaining = total - completed - failed
                st.metric("Remaining", remaining)
    
    def _render_type_progress(self, status: Dict) -> None:
        """Render progress by document type"""
        
        st.subheader("📊 Progress by Document Type")
        
        type_stats = status.get('type_statistics', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            transcript_stats = type_stats.get('transcript', {})
            self._render_type_card("📋 Transcripts", transcript_stats, "blue")
        
        with col2:
            presentation_stats = type_stats.get('presentation', {})
            self._render_type_card("📊 Presentations", presentation_stats, "green")
        
        with col3:
            summary_stats = type_stats.get('financial_summary', {})
            self._render_type_card("📈 Summaries", summary_stats, "orange")
    
    def _render_active_tasks(self, status: Dict) -> None:
        """Render currently active processing tasks"""
        
        active_tasks = status.get('active_tasks', {})
        
        if active_tasks:
            st.subheader("⚡ Active Processing Tasks")
            
            # Create a table of active tasks
            task_data = []
            for task_id, task_info in active_tasks.items():
                task_data.append({
                    'Worker ID': task_info.get('worker_id', 'Unknown'),
                    'Document': task_info.get('document_name', 'Unknown'),
                    'Type': task_info.get('document_type', 'Unknown'),
                    'Progress': f"{task_info.get('progress', 0):.1%}",
                    'Elapsed Time': f"{task_info.get('elapsed_time', 0):.1f}s",
                    'Status': task_info.get('status', 'Processing')
                })
            
            if task_data:
                df = pd.DataFrame(task_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No active processing tasks")
```

---

This comprehensive project structure and technical specification provides the foundation for implementing the multi-document concurrent processing system. The architecture is designed to be scalable, maintainable, and extensible for future enhancements.

The next step would be to begin implementation starting with Phase 1 (Foundation & File Handling), followed by the document classification system, and then the concurrent processing engine.