# 🚀 FinDocPipeline Multi-Document Implementation Roadmap

## 📋 **Implementation Phases**

### **Phase 1: Foundation & File Handling (Week 1-2)**

#### **1.1 Enhanced Requirements & Dependencies**
```python
# New dependencies to add to requirements.txt
python-docx>=0.8.11          # DOCX processing
openpyxl>=3.1.2              # Excel XLSX processing
xlrd>=2.0.1                  # Excel XLS processing
pandas>=2.0.0                # Enhanced data processing
asyncio-throttle>=1.0.2      # Concurrent processing control
aiofiles>=23.1.0             # Async file operations
pydantic>=2.0.0              # Schema validation
spacy>=3.7.0                 # Advanced NLP
transformers>=4.30.0         # AI-powered classification
sentence-transformers>=2.2.2 # Semantic embeddings
```

#### **1.2 Multi-Format File Handlers**
```python
# Create: file_handlers/
├── __init__.py
├── base_handler.py          # Abstract base class
├── pdf_handler.py           # Enhanced PDF processing
├── docx_handler.py          # Microsoft Word processing
├── txt_handler.py           # Plain text processing
├── csv_handler.py           # CSV data processing
├── excel_handler.py         # Excel file processing
└── file_detector.py         # File type detection
```

#### **1.3 Core File Handler Implementation**
```python
# file_handlers/base_handler.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import mimetypes
import magic

class BaseFileHandler(ABC):
    """Abstract base class for all file handlers"""
    
    def __init__(self):
        self.supported_extensions = []
        self.mime_types = []
    
    @abstractmethod
    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """Extract content from file"""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract file metadata"""
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file can be processed"""
        return self._check_file_type(file_path) and self._check_file_size(file_path)
    
    def _check_file_type(self, file_path: str) -> bool:
        """Check if file type is supported"""
        file_ext = file_path.lower().split('.')[-1]
        return file_ext in self.supported_extensions
    
    def _check_file_size(self, file_path: str, max_size_mb: int = 100) -> bool:
        """Check file size limits"""
        import os
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        return file_size <= max_size_mb

# file_handlers/docx_handler.py
from docx import Document
from docx.table import Table
import xml.etree.ElementTree as ET

class DOCXHandler(BaseFileHandler):
    """Microsoft Word document handler"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['docx']
        self.mime_types = ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']
    
    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive content from DOCX"""
        doc = Document(file_path)
        
        content = {
            'text': self._extract_text(doc),
            'paragraphs': self._extract_paragraphs(doc),
            'tables': self._extract_tables(doc),
            'headers_footers': self._extract_headers_footers(doc),
            'styles': self._extract_styles(doc),
            'structure': self._analyze_structure(doc)
        }
        
        return content
    
    def _extract_text(self, doc: Document) -> str:
        """Extract all text content"""
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)
    
    def _extract_tables(self, doc: Document) -> List[Dict]:
        """Extract table data"""
        tables = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            tables.append({
                'data': table_data,
                'rows': len(table.rows),
                'cols': len(table.columns) if table.rows else 0
            })
        return tables
    
    def _analyze_structure(self, doc: Document) -> Dict:
        """Analyze document structure"""
        structure = {
            'total_paragraphs': len(doc.paragraphs),
            'total_tables': len(doc.tables),
            'heading_levels': self._count_heading_levels(doc),
            'has_toc': self._has_table_of_contents(doc)
        }
        return structure

# file_handlers/excel_handler.py
import pandas as pd
import openpyxl
from openpyxl.chart import Chart

class ExcelHandler(BaseFileHandler):
    """Excel file handler for XLSX and XLS"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['xlsx', 'xls']
        self.mime_types = [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel'
        ]
    
    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive Excel content"""
        
        # Load workbook
        if file_path.endswith('.xlsx'):
            wb = openpyxl.load_workbook(file_path, data_only=True)
            wb_formulas = openpyxl.load_workbook(file_path, data_only=False)
        else:
            # Handle XLS files
            wb = pd.ExcelFile(file_path)
        
        content = {
            'worksheets': self._extract_worksheets(wb),
            'charts': self._extract_charts(wb) if hasattr(wb, 'worksheets') else [],
            'named_ranges': self._extract_named_ranges(wb) if hasattr(wb, 'defined_names') else [],
            'metadata': self._extract_excel_metadata(wb),
            'financial_patterns': self._detect_financial_patterns(wb)
        }
        
        return content
    
    def _extract_worksheets(self, wb) -> List[Dict]:
        """Extract data from all worksheets"""
        worksheets = []
        
        if hasattr(wb, 'worksheets'):  # openpyxl workbook
            for ws in wb.worksheets:
                ws_data = {
                    'name': ws.title,
                    'data': self._worksheet_to_dataframe(ws),
                    'dimensions': f"{ws.max_row}x{ws.max_column}",
                    'has_formulas': self._has_formulas(ws),
                    'financial_indicators': self._detect_financial_indicators(ws)
                }
                worksheets.append(ws_data)
        else:  # pandas ExcelFile
            for sheet_name in wb.sheet_names:
                df = pd.read_excel(wb, sheet_name=sheet_name)
                ws_data = {
                    'name': sheet_name,
                    'data': df,
                    'dimensions': f"{len(df)}x{len(df.columns)}",
                    'financial_indicators': self._detect_financial_indicators_df(df)
                }
                worksheets.append(ws_data)
        
        return worksheets
    
    def _detect_financial_patterns(self, wb) -> Dict:
        """Detect common financial statement patterns"""
        patterns = {
            'income_statement': False,
            'balance_sheet': False,
            'cash_flow': False,
            'financial_ratios': False,
            'quarterly_data': False
        }
        
        # Analyze worksheet names and content
        if hasattr(wb, 'worksheets'):
            for ws in wb.worksheets:
                sheet_name = ws.title.lower()
                if any(term in sheet_name for term in ['income', 'p&l', 'profit', 'loss']):
                    patterns['income_statement'] = True
                elif any(term in sheet_name for term in ['balance', 'bs', 'position']):
                    patterns['balance_sheet'] = True
                elif any(term in sheet_name for term in ['cash', 'flow', 'cf']):
                    patterns['cash_flow'] = True
                elif any(term in sheet_name for term in ['ratio', 'kpi', 'metrics']):
                    patterns['financial_ratios'] = True
        
        return patterns
```

---

### **Phase 2: Document Classification & Schema Design (Week 3-4)**

#### **2.1 AI-Powered Document Classifier**
```python
# Create: classification/
├── __init__.py
├── document_classifier.py   # Main classification logic
├── feature_extractors.py    # Content feature extraction
├── models/                  # Pre-trained models
│   ├── transcript_model.pkl
│   ├── presentation_model.pkl
│   └── summary_model.pkl
└── training_data/           # Classification training data
```

#### **2.2 Advanced Classification Implementation**
```python
# classification/document_classifier.py
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

class AdvancedDocumentClassifier:
    """AI-powered document type classification"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.ml_classifier = self._load_ml_classifier()
        self.transformer_model = self._load_transformer_model()
        
        # Classification thresholds
        self.confidence_threshold = 0.7
        self.feature_weights = {
            'structural': 0.3,
            'linguistic': 0.4,
            'semantic': 0.3
        }
    
    def classify_document(self, content: Dict, metadata: Dict) -> Dict[str, Any]:
        """Comprehensive document classification"""
        
        # Extract multiple feature types
        structural_features = self.feature_extractor.extract_structural_features(content)
        linguistic_features = self.feature_extractor.extract_linguistic_features(content)
        semantic_features = self.feature_extractor.extract_semantic_features(content)
        
        # Combine features
        combined_features = self._combine_features(
            structural_features, 
            linguistic_features, 
            semantic_features
        )
        
        # Get predictions from multiple models
        ml_prediction = self.ml_classifier.predict_proba([combined_features])[0]
        transformer_prediction = self._get_transformer_prediction(content['text'])
        
        # Ensemble prediction
        final_prediction = self._ensemble_predictions(ml_prediction, transformer_prediction)
        
        # Determine document type and confidence
        doc_types = ['transcript', 'presentation', 'financial_summary']
        predicted_type = doc_types[np.argmax(final_prediction)]
        confidence = np.max(final_prediction)
        
        return {
            'document_type': predicted_type,
            'confidence': confidence,
            'type_probabilities': dict(zip(doc_types, final_prediction)),
            'features_used': {
                'structural': structural_features,
                'linguistic': linguistic_features,
                'semantic': semantic_features
            }
        }
    
    def _get_transformer_prediction(self, text: str) -> np.ndarray:
        """Get prediction from transformer model"""
        # Use pre-trained financial document classifier
        classifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",  # Or custom trained model
            tokenizer="microsoft/DialoGPT-medium"
        )
        
        # Truncate text for transformer input limits
        truncated_text = text[:512]
        result = classifier(truncated_text)
        
        # Convert to probability distribution
        return self._convert_to_probabilities(result)

# classification/feature_extractors.py
class FeatureExtractor:
    """Extract various features for document classification"""
    
    def extract_structural_features(self, content: Dict) -> List[float]:
        """Extract document structure features"""
        text = content.get('text', '')
        
        features = [
            # Length features
            len(text),
            len(text.split()),
            len(text.split('\n')),
            
            # Speaker/dialogue features
            self._count_speaker_patterns(text),
            self._count_qa_patterns(text),
            self._count_timestamp_patterns(text),
            
            # Presentation features
            self._count_slide_indicators(text),
            self._count_bullet_points(text),
            self._count_chart_references(text),
            
            # Financial features
            self._count_financial_terms(text),
            self._count_numerical_data(text),
            self._count_table_structures(text),
            
            # Document structure
            self._calculate_paragraph_consistency(text),
            self._calculate_section_headers(text)
        ]
        
        return features
    
    def extract_linguistic_features(self, content: Dict) -> List[float]:
        """Extract linguistic and stylistic features"""
        text = content.get('text', '')
        
        features = [
            # Readability metrics
            self._calculate_flesch_reading_ease(text),
            self._calculate_avg_sentence_length(text),
            self._calculate_avg_word_length(text),
            
            # Formality indicators
            self._calculate_formality_score(text),
            self._count_passive_voice(text),
            self._count_technical_terms(text),
            
            # Communication style
            self._count_first_person_pronouns(text),
            self._count_questions(text),
            self._count_exclamations(text),
            
            # Financial language
            self._count_forward_looking_statements(text),
            self._count_risk_language(text),
            self._count_performance_language(text)
        ]
        
        return features
    
    def extract_semantic_features(self, content: Dict) -> List[float]:
        """Extract semantic embeddings and topic features"""
        text = content.get('text', '')
        
        # Generate sentence embeddings
        embeddings = self._generate_embeddings(text)
        
        # Topic modeling features
        topic_features = self._extract_topic_features(text)
        
        # Combine semantic features
        features = embeddings.tolist() + topic_features
        
        return features
```

#### **2.3 Schema Validation System**
```python
# Create: schemas/
├── __init__.py
├── base_schema.py           # Base schema definitions
├── transcript_schema.py     # Transcript-specific schema
├── presentation_schema.py   # Presentation-specific schema
├── summary_schema.py        # Financial summary schema
└── validators.py            # Schema validation logic

# schemas/transcript_schema.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class SpeakerRole(str, Enum):
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    ANALYST = "Analyst"
    MODERATOR = "Moderator"
    INVESTOR = "Investor"
    OTHER = "Other"

class ContentType(str, Enum):
    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"
    OPENING = "opening"
    CLOSING = "closing"
    DISCUSSION = "discussion"

class TopicCategory(str, Enum):
    FINANCIAL_PERFORMANCE = "financial_performance"
    GUIDANCE = "guidance"
    STRATEGY = "strategy"
    RISK_FACTORS = "risk_factors"
    MARKET_CONDITIONS = "market_conditions"
    OPERATIONS = "operations"

class Speaker(BaseModel):
    speaker_id: str = Field(..., description="Unique speaker identifier")
    speaker_name: str = Field(..., description="Full name of speaker")
    speaker_role: SpeakerRole = Field(..., description="Role/title of speaker")
    company: str = Field(..., description="Company affiliation")
    total_speaking_time: Optional[float] = Field(None, description="Total speaking time in minutes")
    segment_count: int = Field(0, description="Number of segments spoken")
    
    @validator('speaker_name')
    def validate_speaker_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Speaker name must be at least 2 characters')
        return v.strip()

class TranscriptSegment(BaseModel):
    segment_id: str = Field(..., description="Unique segment identifier")
    speaker_id: str = Field(..., description="Reference to speaker")
    timestamp: Optional[str] = Field(None, description="Timestamp in format HH:MM:SS")
    sequence_number: int = Field(..., description="Order in transcript")
    content: str = Field(..., description="Spoken content")
    content_type: ContentType = Field(..., description="Type of content")
    topic_classification: TopicCategory = Field(..., description="Primary topic")
    sentiment: str = Field(..., description="Sentiment: positive/negative/neutral")
    key_metrics_mentioned: List[str] = Field(default_factory=list)
    forward_looking: bool = Field(False, description="Contains forward-looking statements")
    chunk_embedding: Optional[List[float]] = Field(None, description="Semantic embedding vector")
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Content must be at least 10 characters')
        return v.strip()
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        if v not in ['positive', 'negative', 'neutral']:
            raise ValueError('Sentiment must be positive, negative, or neutral')
        return v

class QAPair(BaseModel):
    question_id: str = Field(..., description="Unique question identifier")
    analyst_name: str = Field(..., description="Name of analyst asking question")
    analyst_firm: str = Field(..., description="Analyst's firm")
    question_text: str = Field(..., description="Full question text")
    question_topic: TopicCategory = Field(..., description="Question topic category")
    responder_name: str = Field(..., description="Name of person responding")
    response_text: str = Field(..., description="Full response text")
    question_timestamp: Optional[str] = Field(None, description="Question timestamp")
    response_timestamp: Optional[str] = Field(None, description="Response timestamp")

class FinancialMention(BaseModel):
    metric_name: str = Field(..., description="Name of financial metric")
    value: Optional[float] = Field(None, description="Numerical value if extractable")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    period: Optional[str] = Field(None, description="Time period reference")
    context: str = Field(..., description="Surrounding context")
    speaker_id: str = Field(..., description="Speaker who mentioned it")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")

class Topic(BaseModel):
    topic_name: str = Field(..., description="Topic name")
    topic_category: TopicCategory = Field(..., description="Topic category")
    mentions_count: int = Field(..., ge=0, description="Number of mentions")
    speakers_involved: List[str] = Field(..., description="Speakers who discussed topic")
    sentiment_distribution: Dict[str, float] = Field(..., description="Sentiment breakdown")
    key_phrases: List[str] = Field(default_factory=list, description="Key phrases for topic")

class TranscriptMetadata(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type/extension")
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    quarter: Optional[str] = Field(None, description="Financial quarter (e.g., Q1 2024)")
    bank_name: Optional[str] = Field(None, description="Bank/company name")
    company_ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    call_date: Optional[datetime] = Field(None, description="Date of earnings call")
    call_type: str = Field(default="earnings", description="Type of call")
    
    @validator('quarter')
    def validate_quarter(cls, v):
        if v and not re.match(r'^Q[1-4]\s+\d{4}$', v):
            raise ValueError('Quarter must be in format "Q1 2024"')
        return v

class TranscriptSchema(BaseModel):
    """Complete schema for earnings call transcripts"""
    document_metadata: TranscriptMetadata
    speakers: List[Speaker]
    segments: List[TranscriptSegment]
    qa_pairs: List[QAPair]
    financial_mentions: List[FinancialMention]
    topics: List[Topic]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('speakers')
    def validate_speakers_not_empty(cls, v):
        if not v:
            raise ValueError('At least one speaker must be present')
        return v
    
    @validator('segments')
    def validate_segments_not_empty(cls, v):
        if not v:
            raise ValueError('At least one segment must be present')
        return v
```

---

### **Phase 3: Concurrent Processing Engine (Week 5-6)**

#### **3.1 Async Processing Manager**
```python
# Create: processing/
├── __init__.py
├── concurrent_manager.py    # Main concurrent processing
├── task_queue.py           # Task queue management
├── progress_tracker.py     # Progress tracking
├── resource_manager.py     # Resource allocation
└── result_aggregator.py    # Results consolidation

# processing/concurrent_manager.py
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Callable
import time
import logging
from dataclasses import dataclass
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingTask:
    task_id: str
    document_path: str
    document_type: str
    file_size: int
    priority: int = 1
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class ConcurrentDocumentProcessor:
    """Advanced concurrent document processing manager"""
    
    def __init__(self, 
                 max_workers: int = 50,
                 max_memory_mb: int = 8192,
                 enable_gpu: bool = False):
        
        self.max_workers = max_workers
        self.max_memory_mb = max_memory_mb
        self.enable_gpu = enable_gpu
        
        # Processing components
        self.task_queue = TaskQueue()
        self.progress_tracker = ProgressTracker()
        self.resource_manager = ResourceManager(max_memory_mb)
        self.result_aggregator = ResultAggregator()
        
        # Executors for different types of processing
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(max_workers, 8))
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'average_processing_time': 0.0,
            'peak_memory_usage': 0.0
        }
        
        # Active tasks tracking
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingTask] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def process_document_batch(self, 
                                   documents: List[Dict],
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process multiple documents concurrently with advanced management"""
        
        batch_start_time = time.time()
        
        # Initialize batch processing
        batch_id = f"batch_{int(batch_start_time)}"
        self.logger.info(f"Starting batch processing: {batch_id} with {len(documents)} documents")
        
        # Create processing tasks
        tasks = []
        for i, doc in enumerate(documents):
            task = ProcessingTask(
                task_id=f"{batch_id}_task_{i}",
                document_path=doc['path'],
                document_type=doc.get('type', 'unknown'),
                file_size=doc.get('size', 0),
                priority=doc.get('priority', 1)
            )
            tasks.append(task)
        
        # Initialize progress tracking
        self.progress_tracker.initialize_batch(batch_id, tasks)
        
        # Process tasks with resource management
        results = await self._process_tasks_with_resource_management(
            tasks, progress_callback
        )
        
        # Aggregate and analyze results
        batch_results = self.result_aggregator.aggregate_batch_results(
            batch_id, results, time.time() - batch_start_time
        )
        
        # Update statistics
        self._update_batch_statistics(batch_results)
        
        self.logger.info(f"Batch processing completed: {batch_id}")
        return batch_results
    
    async def _process_tasks_with_resource_management(self,
                                                    tasks: List[ProcessingTask],
                                                    progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Process tasks with intelligent resource management"""
        
        # Sort tasks by priority and size
        sorted_tasks = sorted(tasks, key=lambda t: (t.priority, -t.file_size))
        
        # Create semaphore for concurrent processing control
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Process tasks concurrently
        async def process_single_task(task: ProcessingTask) -> Dict:
            async with semaphore:
                return await self._process_single_task_async(task, progress_callback)
        
        # Execute all tasks
        results = await asyncio.gather(
            *[process_single_task(task) for task in sorted_tasks],
            return_exceptions=True
        )
        
        # Handle exceptions and convert to results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'task_id': sorted_tasks[i].task_id,
                    'status': ProcessingStatus.FAILED,
                    'error': str(result),
                    'processing_time': 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_task_async(self,
                                       task: ProcessingTask,
                                       progress_callback: Optional[Callable] = None) -> Dict:
        """Process individual task asynchronously"""
        
        task.started_at = time.time()
        task.status = ProcessingStatus.PROCESSING
        self.active_tasks[task.task_id] = task
        
        try:
            # Check resource availability
            if not self.resource_manager.can_process_task(task):
                await self.resource_manager.wait_for_resources(task)
            
            # Allocate resources
            self.resource_manager.allocate_resources(task)
            
            # Determine processing strategy based on file type and size
            if task.file_size > 50 * 1024 * 1024:  # 50MB
                # Use process executor for large files
                result = await self._process_with_process_executor(task)
            else:
                # Use thread executor for smaller files
                result = await self._process_with_thread_executor(task)
            
            # Update task status
            task.completed_at = time.time()
            task.status = ProcessingStatus.COMPLETED
            task.result = result
            
            # Update progress
            if progress_callback:
                await progress_callback(task)
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            del self.active_tasks[task.task_id]
            
            return {
                'task_id': task.task_id,
                'status': ProcessingStatus.COMPLETED,
                'result': result,
                'processing_time': task.completed_at - task.started_at
            }
            
        except Exception as e:
            # Handle processing error
            task.completed_at = time.time()
            task.status = ProcessingStatus.FAILED
            task.error = str(e)
            
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            return {
                'task_id': task.task_id,
                'status': ProcessingStatus.FAILED,
                'error': str(e),
                'processing_time': task.completed_at - task.started_at if task.completed_at else 0.0
            }
        
        finally:
            # Release resources
            self.resource_manager.release_resources(task)
    
    async def _process_with_thread_executor(self, task: ProcessingTask) -> Dict:
        """Process task using thread executor"""
        loop = asyncio.get_event_loop()
        
        # Run in thread executor
        result = await loop.run_in_executor(
            self.thread_executor,
            self._process_document_sync,
            task.document_path,
            task.document_type
        )
        
        return result
    
    async def _process_with_process_executor(self, task: ProcessingTask) -> Dict:
        """Process task using process executor for CPU-intensive work"""
        loop = asyncio.get_event_loop()
        
        # Run in process executor
        result = await loop.run_in_executor(
            self.process_executor,
            self._process_document_sync,
            task.document_path,
            task.document_type
        )
        
        return result
    
    def _process_document_sync(self, document_path: str, document_type: str) -> Dict:
        """Synchronous document processing (runs in executor)"""
        
        # Import processing components (inside function to avoid pickling issues)
        from file_handlers import UniversalFileHandler
        from classification import AdvancedDocumentClassifier
        from processing.type_processors import (
            TranscriptProcessor, 
            PresentationProcessor, 
            FinancialSummaryProcessor