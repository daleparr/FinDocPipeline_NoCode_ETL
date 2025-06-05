"""
Base file handler for multi-format document processing.
Provides abstract interface for all file type handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import streamlit as st

class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"

@dataclass
class ExtractedContent:
    """Standardized content structure from file processing"""
    text: str
    metadata: Dict[str, Any]
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    structure: Dict[str, Any]
    raw_data: Optional[Any] = None
    processing_time: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class BaseFileHandler(ABC):
    """Abstract base class for all file handlers"""
    
    def __init__(self):
        self.supported_extensions = []
        self.handler_name = self.__class__.__name__
    
    @abstractmethod
    def extract_content(self, uploaded_file) -> ExtractedContent:
        """
        Extract content from uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            ExtractedContent: Standardized content structure
        """
        pass
    
    @abstractmethod
    def validate_file(self, uploaded_file) -> bool:
        """
        Validate if file can be processed by this handler.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            bool: True if file can be processed
        """
        pass
    
    def get_file_info(self, uploaded_file) -> Dict[str, Any]:
        """
        Get basic file information.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Dict containing file metadata
        """
        return {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'handler': self.handler_name
        }
    
    def _create_error_content(self, error_message: str, filename: str) -> ExtractedContent:
        """
        Create error content structure when processing fails.
        
        Args:
            error_message: Error description
            filename: Name of the file that failed
            
        Returns:
            ExtractedContent with error information
        """
        return ExtractedContent(
            text="",
            metadata={'filename': filename, 'error': True},
            tables=[],
            images=[],
            structure={},
            errors=[error_message]
        )
    
    def _extract_basic_metadata(self, uploaded_file) -> Dict[str, Any]:
        """
        Extract basic metadata common to all file types.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Dict containing basic metadata
        """
        return {
            'filename': uploaded_file.name,
            'size_bytes': uploaded_file.size,
            'size_mb': round(uploaded_file.size / (1024 * 1024), 2),
            'file_type': uploaded_file.type,
            'handler_used': self.handler_name,
            'supported_extensions': self.supported_extensions
        }
    
    @st.cache_data
    def _cached_file_info(_self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Cached file information extraction to improve performance.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Dict containing cached file information
        """
        return {
            'size': len(file_bytes),
            'filename': filename,
            'cached_at': st.session_state.get('current_time', 'unknown')
        }
    
    def _estimate_processing_time(self, file_size_bytes: int) -> float:
        """
        Estimate processing time based on file size.
        
        Args:
            file_size_bytes: File size in bytes
            
        Returns:
            Estimated processing time in seconds
        """
        # Base processing time + size-based scaling
        base_time = 2.0  # 2 seconds base
        size_mb = file_size_bytes / (1024 * 1024)
        
        # Different scaling factors for different file types
        scaling_factors = {
            'pdf': 1.5,
            'docx': 1.0,
            'xlsx': 2.0,
            'csv': 0.5,
            'txt': 0.3
        }
        
        # Default scaling factor
        scaling_factor = scaling_factors.get(
            self.supported_extensions[0] if self.supported_extensions else 'pdf',
            1.0
        )
        
        estimated_time = base_time + (size_mb * scaling_factor)
        return min(estimated_time, 30.0)  # Cap at 30 seconds
    
    def _validate_file_size(self, uploaded_file, max_size_mb: int = 10) -> bool:
        """
        Validate file size is within acceptable limits.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            max_size_mb: Maximum allowed file size in MB
            
        Returns:
            bool: True if file size is acceptable
        """
        file_size_mb = uploaded_file.size / (1024 * 1024)
        return file_size_mb <= max_size_mb
    
    def _extract_file_extension(self, filename: str) -> str:
        """
        Extract file extension from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            File extension in lowercase
        """
        return filename.split('.')[-1].lower() if '.' in filename else ''