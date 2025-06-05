# 📊 Enhanced Schema Design for Multi-Document Processing

## 🎯 **Schema Architecture Overview**

The enhanced schema system provides type-specific data structures with rich metadata extraction tailored for financial documents. Each document type has specialized fields while maintaining a common base structure for unified processing.

---

## 🏗️ **Base Schema Foundation**

### **Common Document Schema**
```python
# File: schemas/base_schema.py
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    TRANSCRIPT = "transcript"
    PRESENTATION = "presentation"
    FINANCIAL_SUMMARY = "financial_summary"
    DATA_FILE = "data_file"
    UNKNOWN = "unknown"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BaseDocumentSchema(BaseModel):
    """Base schema for all document types"""
    
    # Core identification
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    document_type: DocumentType = Field(..., description="Classified document type")
    
    # Processing metadata
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    processing_time: Optional[float] = Field(None, description="Processing duration in seconds")
    
    # Classification metadata
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    classification_method: str = Field(..., description="Classification method used")
    
    # File metadata
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File extension")
    
    # Content structure
    text_content: str = Field(..., description="Extracted text content")
    word_count: int = Field(..., description="Total word count")
    page_count: Optional[int] = Field(None, description="Number of pages")
    
    # Tables and structured data
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    images: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted images")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    @validator('word_count', pre=True, always=True)
    def calculate_word_count(cls, v, values):
        if 'text_content' in values:
            return len(values['text_content'].split())
        return v or 0
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

---

## 📋 **Transcript Schema (Earnings Calls)**

### **Transcript-Specific Schema**
```python
# File: schemas/transcript_schema.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import re

class Speaker(BaseModel):
    """Individual speaker in transcript"""
    name: str = Field(..., description="Speaker name")
    title: Optional[str] = Field(None, description="Speaker title/role")
    company: Optional[str] = Field(None, description="Speaker company")
    speaker_type: str = Field(..., description="Type: operator, executive, analyst")
    
class TranscriptSegment(BaseModel):
    """Individual segment of transcript"""
    segment_id: str = Field(..., description="Unique segment identifier")
    speaker: Speaker = Field(..., description="Speaker information")
    content: str = Field(..., description="Segment content")
    timestamp: Optional[str] = Field(None, description="Timestamp if available")
    word_count: int = Field(..., description="Segment word count")
    
    @validator('word_count', pre=True, always=True)
    def calculate_word_count(cls, v, values):
        if 'content' in values:
            return len(values['content'].split())
        return v or 0

class QASection(BaseModel):
    """Question and Answer section"""
    question_id: str = Field(..., description="Question identifier")
    analyst: Speaker = Field(..., description="Analyst asking question")
    question: str = Field(..., description="Question content")
    respondent: Speaker = Field(..., description="Executive responding")
    answer: str = Field(..., description="Answer content")
    follow_up_questions: List[str] = Field(default_factory=list, description="Follow-up questions")

class FinancialMention(BaseModel):
    """Financial data mentioned in transcript"""
    metric: str = Field(..., description="Financial metric name")
    value: Optional[str] = Field(None, description="Mentioned value")
    context: str = Field(..., description="Context around mention")
    segment_id: str = Field(..., description="Segment where mentioned")

class TranscriptSchema(BaseDocumentSchema):
    """Enhanced schema for earnings call transcripts"""
    
    # Company and call information
    company_name: str = Field(..., description="Company name")
    ticker_symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    quarter: Optional[str] = Field(None, description="Reporting quarter (Q1, Q2, Q3, Q4)")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year")
    call_date: Optional[date] = Field(None, description="Call date")
    call_time: Optional[str] = Field(None, description="Call time")
    
    # Transcript structure
    segments: List[TranscriptSegment] = Field(default_factory=list, description="Transcript segments")
    speakers: List[Speaker] = Field(default_factory=list, description="All speakers")
    
    # Q&A Analysis
    qa_sections: List[QASection] = Field(default_factory=list, description="Q&A sections")
    total_questions: int = Field(default=0, description="Total number of questions")
    
    # Financial mentions
    financial_mentions: List[FinancialMention] = Field(default_factory=list, description="Financial data mentions")
    
    # Topic classification
    topics: List[str] = Field(default_factory=list, description="Identified topics")
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Overall sentiment")
    
    # Analyst information
    participating_analysts: List[Speaker] = Field(default_factory=list, description="Analysts who asked questions")
    analyst_firms: List[str] = Field(default_factory=list, description="Analyst firms represented")
    
    # Executive information
    executives_present: List[Speaker] = Field(default_factory=list, description="Company executives present")
    
    # Content analysis
    presentation_duration: Optional[int] = Field(None, description="Presentation duration in minutes")
    qa_duration: Optional[int] = Field(None, description="Q&A duration in minutes")
    
    @validator('quarter')
    def validate_quarter(cls, v):
        if v and v not in ['Q1', 'Q2', 'Q3', 'Q4']:
            raise ValueError('Quarter must be Q1, Q2, Q3, or Q4')
        return v
    
    @validator('fiscal_year')
    def validate_fiscal_year(cls, v):
        if v and (v < 2000 or v > 2030):
            raise ValueError('Fiscal year must be between 2000 and 2030')
        return v
```

### **Transcript Processing Logic**
```python
# File: processing/transcript_processor.py
import streamlit as st
import re
from typing import List, Dict, Any
from schemas.transcript_schema import TranscriptSchema, Speaker, TranscriptSegment, QASection

class TranscriptProcessor:
    """Specialized processor for earnings call transcripts"""
    
    def __init__(self):
        self.speaker_patterns = self._compile_speaker_patterns()
        self.financial_patterns = self._compile_financial_patterns()
    
    def process_transcript(self, content: Dict[str, Any]) -> TranscriptSchema:
        """Process transcript content into structured schema"""
        
        text = content.get('text', '')
        metadata = content.get('metadata', {})
        
        # Extract basic information
        company_info = self._extract_company_info(text, metadata)
        
        # Parse speakers and segments
        segments = self._parse_segments(text)
        speakers = self._extract_speakers(segments)
        
        # Analyze Q&A sections
        qa_sections = self._extract_qa_sections(segments)
        
        # Extract financial mentions
        financial_mentions = self._extract_financial_mentions(segments)
        
        # Topic analysis
        topics = self._analyze_topics(text)
        
        # Create schema
        return TranscriptSchema(
            document_id=self._generate_document_id(metadata),
            filename=metadata.get('filename', ''),
            document_type='transcript',
            classification_confidence=0.95,
            classification_method='transcript_processor',
            file_size=metadata.get('size', 0),
            file_type=metadata.get('type', ''),
            text_content=text,
            
            # Transcript-specific fields
            company_name=company_info.get('company_name', ''),
            ticker_symbol=company_info.get('ticker_symbol'),
            quarter=company_info.get('quarter'),
            fiscal_year=company_info.get('fiscal_year'),
            call_date=company_info.get('call_date'),
            
            segments=segments,
            speakers=speakers,
            qa_sections=qa_sections,
            total_questions=len(qa_sections),
            financial_mentions=financial_mentions,
            topics=topics,
            
            participating_analysts=self._get_analysts(speakers),
            analyst_firms=self._get_analyst_firms(speakers),
            executives_present=self._get_executives(speakers)
        )
    
    def _parse_segments(self, text: str) -> List[TranscriptSegment]:
        """Parse text into speaker segments"""
        
        segments = []
        
        # Split by speaker patterns
        speaker_pattern = r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[:\-]\s*(.+?)(?=^[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[:\-]|\Z)'
        
        matches = re.finditer(speaker_pattern, text, re.MULTILINE | re.DOTALL)
        
        for i, match in enumerate(matches):
            speaker_name = match.group(1).strip()
            content = match.group(2).strip()
            
            # Determine speaker type
            speaker_type = self._classify_speaker_type(speaker_name, content)
            
            # Create speaker
            speaker = Speaker(
                name=speaker_name,
                speaker_type=speaker_type,
                title=self._extract_speaker_title(content),
                company=self._extract_speaker_company(content)
            )
            
            # Create segment
            segment = TranscriptSegment(
                segment_id=f"segment_{i+1}",
                speaker=speaker,
                content=content,
                timestamp=self._extract_timestamp(content)
            )
            
            segments.append(segment)
        
        return segments
    
    def _extract_company_info(self, text: str, metadata: Dict) -> Dict[str, Any]:
        """Extract company and call information"""
        
        info = {}
        
        # Company name patterns
        company_patterns = [
            r'([A-Z][a-zA-Z\s&]+(?:Inc|Corp|Corporation|Company|Ltd))',
            r'Welcome to the ([A-Z][a-zA-Z\s&]+) (?:earnings|quarterly)',
            r'([A-Z][a-zA-Z\s&]+) (?:Q[1-4]|first|second|third|fourth) quarter'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['company_name'] = match.group(1).strip()
                break
        
        # Quarter and year
        quarter_match = re.search(r'(Q[1-4]|first|second|third|fourth) quarter (\d{4})', text, re.IGNORECASE)
        if quarter_match:
            quarter_text = quarter_match.group(1).lower()
            quarter_map = {'first': 'Q1', 'second': 'Q2', 'third': 'Q3', 'fourth': 'Q4'}
            info['quarter'] = quarter_map.get(quarter_text, quarter_text.upper())
            info['fiscal_year'] = int(quarter_match.group(2))
        
        # Ticker symbol
        ticker_match = re.search(r'\b([A-Z]{2,5})\b.*(?:earnings|quarterly)', text)
        if ticker_match:
            info['ticker_symbol'] = ticker_match.group(1)
        
        return info
    
    def _extract_qa_sections(self, segments: List[TranscriptSegment]) -> List[QASection]:
        """Extract Q&A sections from segments"""
        
        qa_sections = []
        current_question = None
        current_answer = None
        
        for segment in segments:
            if segment.speaker.speaker_type == 'analyst':
                # This is likely a question
                if current_question and current_answer:
                    # Save previous Q&A
                    qa_sections.append(QASection(
                        question_id=f"qa_{len(qa_sections)+1}",
                        analyst=current_question['speaker'],
                        question=current_question['content'],
                        respondent=current_answer['speaker'],
                        answer=current_answer['content']
                    ))
                
                # Start new question
                current_question = {
                    'speaker': segment.speaker,
                    'content': segment.content
                }
                current_answer = None
            
            elif segment.speaker.speaker_type == 'executive' and current_question:
                # This is likely an answer
                current_answer = {
                    'speaker': segment.speaker,
                    'content': segment.content
                }
        
        # Don't forget the last Q&A
        if current_question and current_answer:
            qa_sections.append(QASection(
                question_id=f"qa_{len(qa_sections)+1}",
                analyst=current_question['speaker'],
                question=current_question['content'],
                respondent=current_answer['speaker'],
                answer=current_answer['content']
            ))
        
        return qa_sections
    
    def _classify_speaker_type(self, name: str, content: str) -> str:
        """Classify speaker type based on name and content"""
        
        name_lower = name.lower()
        content_lower = content.lower()
        
        # Operator indicators
        if 'operator' in name_lower:
            return 'operator'
        
        # Executive indicators
        executive_titles = ['ceo', 'cfo', 'president', 'chairman', 'chief', 'director']
        if any(title in content_lower[:100] for title in executive_titles):
            return 'executive'
        
        # Analyst indicators
        analyst_indicators = ['analyst', 'research', 'securities', 'capital', 'investment']
        if any(indicator in content_lower[:100] for indicator in analyst_indicators):
            return 'analyst'
        
        # Default classification based on content patterns
        if any(phrase in content_lower for phrase in ['thank you', 'next question', 'that concludes']):
            return 'operator'
        elif any(phrase in content_lower for phrase in ['my question', 'i wanted to ask', 'could you']):
            return 'analyst'
        else:
            return 'executive'
```

---

## 📊 **Presentation Schema (Slide Decks)**

### **Presentation-Specific Schema**
```python
# File: schemas/presentation_schema.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import date

class Slide(BaseModel):
    """Individual slide information"""
    slide_number: int = Field(..., description="Slide number")
    title: Optional[str] = Field(None, description="Slide title")
    content: str = Field(..., description="Slide content")
    bullet_points: List[str] = Field(default_factory=list, description="Bullet points")
    charts_mentioned: List[str] = Field(default_factory=list, description="Charts/graphs mentioned")
    financial_data: List[Dict[str, Any]] = Field(default_factory=list, description="Financial data on slide")

class FinancialHighlight(BaseModel):
    """Financial highlight from presentation"""
    metric: str = Field(..., description="Financial metric")
    value: str = Field(..., description="Value")
    period: Optional[str] = Field(None, description="Time period")
    comparison: Optional[str] = Field(None, description="Comparison (YoY, QoQ, etc.)")
    slide_number: int = Field(..., description="Source slide number")

class PresentationSchema(BaseDocumentSchema):
    """Enhanced schema for financial presentations"""
    
    # Presentation metadata
    company_name: str = Field(..., description="Company name")
    presentation_title: str = Field(..., description="Presentation title")
    presentation_date: Optional[date] = Field(None, description="Presentation date")
    quarter: Optional[str] = Field(None, description="Reporting quarter")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year")
    
    # Slide structure
    slides: List[Slide] = Field(default_factory=list, description="Individual slides")
    total_slides: int = Field(default=0, description="Total number of slides")
    
    # Content analysis
    agenda_items: List[str] = Field(default_factory=list, description="Agenda items")
    key_highlights: List[str] = Field(default_factory=list, description="Key highlights")
    financial_highlights: List[FinancialHighlight] = Field(default_factory=list, description="Financial highlights")
    
    # Structure analysis
    has_agenda: bool = Field(default=False, description="Has agenda slide")
    has_financial_summary: bool = Field(default=False, description="Has financial summary")
    has_outlook: bool = Field(default=False, description="Has outlook/guidance")
    
    # Chart and visual analysis
    charts_count: int = Field(default=0, description="Number of charts mentioned")
    tables_count: int = Field(default=0, description="Number of tables")
    
    @validator('total_slides', pre=True, always=True)
    def calculate_total_slides(cls, v, values):
        if 'slides' in values:
            return len(values['slides'])
        return v or 0
```

---

## 📈 **Financial Summary Schema**

### **Financial Summary-Specific Schema**
```python
# File: schemas/summary_schema.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import date
from decimal import Decimal

class FinancialStatement(BaseModel):
    """Financial statement data"""
    statement_type: str = Field(..., description="Type: income_statement, balance_sheet, cash_flow")
    period: str = Field(..., description="Period (Q1 2024, FY 2023, etc.)")
    line_items: Dict[str, Union[str, float, None]] = Field(..., description="Financial line items")

class FinancialRatio(BaseModel):
    """Financial ratio calculation"""
    ratio_name: str = Field(..., description="Ratio name")
    value: Optional[float] = Field(None, description="Ratio value")
    period: str = Field(..., description="Period")
    benchmark: Optional[float] = Field(None, description="Industry benchmark")

class FinancialSummarySchema(BaseDocumentSchema):
    """Enhanced schema for financial summary documents"""
    
    # Company information
    company_name: str = Field(..., description="Company name")
    ticker_symbol: Optional[str] = Field(None, description="Stock ticker")
    reporting_period: str = Field(..., description="Reporting period")
    fiscal_year: int = Field(..., description="Fiscal year")
    
    # Financial statements
    income_statement: Optional[FinancialStatement] = Field(None, description="Income statement data")
    balance_sheet: Optional[FinancialStatement] = Field(None, description="Balance sheet data")
    cash_flow_statement: Optional[FinancialStatement] = Field(None, description="Cash flow data")
    
    # Key metrics
    revenue: Optional[float] = Field(None, description="Total revenue")
    net_income: Optional[float] = Field(None, description="Net income")
    eps: Optional[float] = Field(None, description="Earnings per share")
    
    # Financial ratios
    financial_ratios: List[FinancialRatio] = Field(default_factory=list, description="Calculated ratios")
    
    # Analysis
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Key financial metrics")
    trends: List[str] = Field(default_factory=list, description="Identified trends")
    
    # Data quality
    data_completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Data completeness score")
    has_comparative_data: bool = Field(default=False, description="Has comparative period data")
```

---

## 🔄 **Schema Processing Pipeline**

### **Unified Schema Processor**
```python
# File: processing/schema_processor.py
import streamlit as st
from typing import Dict, Any, Union
from schemas.base_schema import BaseDocumentSchema, DocumentType
from schemas.transcript_schema import TranscriptSchema
from schemas.presentation_schema import PresentationSchema
from schemas.summary_schema import FinancialSummarySchema
from processing.transcript_processor import TranscriptProcessor
from processing.presentation_processor import PresentationProcessor
from processing.summary_processor import SummaryProcessor

class UnifiedSchemaProcessor:
    """Unified processor for all document types"""
    
    def __init__(self):
        self.processors = {
            DocumentType.TRANSCRIPT: TranscriptProcessor(),
            DocumentType.PRESENTATION: PresentationProcessor(),
            DocumentType.FINANCIAL_SUMMARY: SummaryProcessor()
        }
    
    def process_document(
        self, 
        content: Dict[str, Any], 
        document_type: DocumentType
    ) -> Union[TranscriptSchema, PresentationSchema, FinancialSummarySchema]:
        """Process document based on type"""
        
        processor = self.processors.get(document_type)
        
        if not processor:
            # Fallback to base schema
            return self._create_base_schema(content, document_type)
        
        try:
            return processor.process(content)
        except Exception as e:
            st.error(f"Error processing {document_type}: {str(e)}")
            return self._create_base_schema(content, document_type)
    
    def _create_base_schema(self, content: Dict[str, Any], doc_type: DocumentType) -> BaseDocumentSchema:
        """Create base schema as fallback"""
        
        metadata = content.get('metadata', {})
        
        return BaseDocumentSchema(
            document_id=self._generate_id(metadata),
            filename=metadata.get('filename', ''),
            document_type=doc_type,
            classification_confidence=0.5,
            classification_method='fallback',
            file_size=metadata.get('size', 0),
            file_type=metadata.get('type', ''),
            text_content=content.get('text', ''),
            tables=content.get('tables', []),
            images=content.get('images', [])
        )
    
    @st.cache_data
    def validate_schema(_self, schema_data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """Validate schema data"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Type-specific validation
        if schema_type == 'transcript':
            validation_result = _self._validate_transcript_schema(schema_data)
        elif schema_type == 'presentation':
            validation_result = _self._validate_presentation_schema(schema_data)
        elif schema_type == 'financial_summary':
            validation_result = _self._validate_summary_schema(schema_data)
        
        return validation_result
    
    def _validate_transcript_schema(self, data: Dict) -> Dict[str, Any]:
        """Validate transcript schema"""
        
        errors = []
        warnings = []
        
        # Check required fields
        if not data.get('company_name'):
            errors.append("Company name is required for transcripts")
        
        if not data.get('segments'):
            warnings.append("No speaker segments found")
        
        if not data.get('qa_sections'):
            warnings.append("No Q&A sections identified")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
```

This enhanced schema design provides comprehensive data structures for each document type while maintaining consistency and enabling rich metadata extraction. The schemas are designed to work seamlessly with Streamlit's caching and session state management.

Would you like me to create the final execution roadmap and timeline for implementing this multi-document processing system?