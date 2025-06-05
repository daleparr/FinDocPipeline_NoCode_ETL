"""
Base schema definitions for enhanced document processing.
Compatible with existing FinDocPipeline data structures.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Document type enumeration"""
    TRANSCRIPT = "transcript"
    PRESENTATION = "presentation"
    FINANCIAL_SUMMARY = "financial_summary"
    UNKNOWN = "unknown"

class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class BaseDocumentSchema:
    """
    Base schema for document processing results.
    Compatible with existing FinDocPipeline output structures.
    """
    
    def __init__(
        self,
        document_id: str,
        filename: str,
        document_type: str,
        file_size: int,
        file_type: str,
        processing_status: str = "pending",
        **kwargs
    ):
        # Core identification
        self.document_id = document_id
        self.filename = filename
        self.document_type = document_type
        
        # File metadata
        self.file_size = file_size
        self.file_type = file_type
        
        # Processing metadata
        self.processing_status = processing_status
        self.processed_at = kwargs.get('processed_at')
        self.processing_time = kwargs.get('processing_time', 0.0)
        
        # Classification metadata
        self.classification_confidence = kwargs.get('classification_confidence', 0.0)
        self.classification_method = kwargs.get('classification_method', 'unknown')
        
        # Content data (compatible with existing FinDocPipeline structures)
        self.pages_data = kwargs.get('pages_data', [])
        self.nlp_data = kwargs.get('nlp_data', [])
        self.visual_data = kwargs.get('visual_data', [])
        self.metrics_data = kwargs.get('metrics_data', [])
        
        # Summary statistics
        self.total_pages = kwargs.get('total_pages', 0)
        self.total_text_length = kwargs.get('total_text_length', 0)
        self.total_tables = kwargs.get('total_tables', 0)
        self.total_images = kwargs.get('total_images', 0)
        self.total_charts = kwargs.get('total_charts', 0)
        self.total_metrics = kwargs.get('total_metrics', 0)
        
        # Error handling
        self.errors = kwargs.get('errors', [])
        self.warnings = kwargs.get('warnings', [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary format"""
        
        return {
            # Core identification
            'document_id': self.document_id,
            'filename': self.filename,
            'document_type': self.document_type,
            
            # File metadata
            'file_size': self.file_size,
            'file_type': self.file_type,
            
            # Processing metadata
            'processing_status': self.processing_status,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processing_time': self.processing_time,
            
            # Classification metadata
            'classification_confidence': self.classification_confidence,
            'classification_method': self.classification_method,
            
            # Content data
            'pages_data': self.pages_data,
            'nlp_data': self.nlp_data,
            'visual_data': self.visual_data,
            'metrics_data': self.metrics_data,
            
            # Summary statistics
            'total_pages': self.total_pages,
            'total_text_length': self.total_text_length,
            'total_tables': self.total_tables,
            'total_images': self.total_images,
            'total_charts': self.total_charts,
            'total_metrics': self.total_metrics,
            
            # Error handling
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    @classmethod
    def from_processing_result(cls, filename: str, processing_result: Dict[str, Any]) -> 'BaseDocumentSchema':
        """
        Create schema from existing FinDocPipeline processing result.
        
        Args:
            filename: Original filename
            processing_result: Result from FinDocPipeline processing
            
        Returns:
            BaseDocumentSchema instance
        """
        
        # Extract basic information
        document_id = f"doc_{hash(filename)}_{int(datetime.now().timestamp())}"
        
        # Extract content analysis
        content_analysis = processing_result.get('content_analysis', {})
        
        # Create schema instance
        return cls(
            document_id=document_id,
            filename=filename,
            document_type=processing_result.get('classification', {}).get('document_type', 'unknown'),
            file_size=processing_result.get('file_size', 0),
            file_type=processing_result.get('file_type', ''),
            processing_status=processing_result.get('status', 'completed'),
            processed_at=datetime.fromisoformat(processing_result['processed_at']) if 'processed_at' in processing_result else datetime.now(),
            processing_time=processing_result.get('processing_time', 0.0),
            classification_confidence=processing_result.get('classification', {}).get('confidence', 0.0),
            classification_method=processing_result.get('classification', {}).get('method', 'unknown'),
            
            # Content data from FinDocPipeline
            pages_data=content_analysis.get('raw_data', {}).get('pages_data', []),
            nlp_data=content_analysis.get('raw_data', {}).get('nlp_dataframe', []),
            visual_data=content_analysis.get('raw_data', {}).get('visual_data', []),
            metrics_data=content_analysis.get('raw_data', {}).get('metrics_data', []),
            
            # Summary statistics
            total_pages=content_analysis.get('pages_processed', 1),
            total_text_length=content_analysis.get('total_text_length', content_analysis.get('text_length', 0)),
            total_tables=content_analysis.get('tables_found', 0),
            total_images=content_analysis.get('images_found', 0),
            total_charts=content_analysis.get('charts_detected', 0),
            total_metrics=content_analysis.get('financial_metrics_extracted', 0),
            
            # Error handling
            errors=[processing_result.get('error')] if processing_result.get('error') else [],
            warnings=[]
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about the document"""
        
        return {
            'filename': self.filename,
            'document_type': self.document_type,
            'processing_status': self.processing_status,
            'file_size_mb': round(self.file_size / (1024 * 1024), 2),
            'processing_time': round(self.processing_time, 2),
            'classification_confidence': round(self.classification_confidence, 2),
            'content_summary': {
                'pages': self.total_pages,
                'text_length': self.total_text_length,
                'tables': self.total_tables,
                'images': self.total_images,
                'charts': self.total_charts,
                'metrics': self.total_metrics
            },
            'has_errors': len(self.errors) > 0,
            'has_warnings': len(self.warnings) > 0
        }
    
    def validate(self) -> Dict[str, Any]:
        """Validate schema data"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required field validation
        if not self.document_id:
            validation_result['errors'].append("Document ID is required")
            validation_result['is_valid'] = False
        
        if not self.filename:
            validation_result['errors'].append("Filename is required")
            validation_result['is_valid'] = False
        
        # Data consistency validation
        if self.processing_status == 'completed' and not self.processed_at:
            validation_result['warnings'].append("Completed document should have processed_at timestamp")
        
        if self.total_pages == 0 and self.processing_status == 'completed':
            validation_result['warnings'].append("Completed document should have at least one page")
        
        if self.classification_confidence < 0.5:
            validation_result['warnings'].append("Low classification confidence")
        
        return validation_result
    
    def add_error(self, error_message: str):
        """Add an error message"""
        if error_message not in self.errors:
            self.errors.append(error_message)
    
    def add_warning(self, warning_message: str):
        """Add a warning message"""
        if warning_message not in self.warnings:
            self.warnings.append(warning_message)
    
    def update_processing_status(self, status: str, processing_time: Optional[float] = None):
        """Update processing status"""
        self.processing_status = status
        
        if status == 'completed':
            self.processed_at = datetime.now()
        
        if processing_time is not None:
            self.processing_time = processing_time