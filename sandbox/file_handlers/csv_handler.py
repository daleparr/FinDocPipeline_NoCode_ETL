"""
CSV file handler for financial data processing.
"""

import streamlit as st
import pandas as pd
import io
import time
from typing import Dict, Any, List
from .base_handler import BaseFileHandler, ExtractedContent

class CSVHandler(BaseFileHandler):
    """CSV file handler for financial data"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['csv']
        self.handler_name = "CSVHandler"
    
    def validate_file(self, uploaded_file) -> bool:
        """Validate CSV file"""
        try:
            if not uploaded_file.name.lower().endswith('.csv'):
                return False
            
            if not self._validate_file_size(uploaded_file, max_size_mb=5):
                return False
            
            return True
        except Exception:
            return False
    
    def extract_content(self, uploaded_file) -> ExtractedContent:
        """Extract content from CSV file"""
        start_time = time.time()
        
        try:
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            content = self._extract_csv_cached(file_bytes, uploaded_file.name)
            content.processing_time = time.time() - start_time
            
            return content
        
        except Exception as e:
            return self._create_error_content(
                f"CSV processing error: {str(e)}", 
                uploaded_file.name
            )
    
    @st.cache_data
    def _extract_csv_cached(_self, file_bytes: bytes, filename: str) -> ExtractedContent:
        """Cached CSV extraction"""
        
        try:
            csv_file = io.BytesIO(file_bytes)
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    csv_file.seek(0)
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("Could not decode CSV file")
            
            # Convert to text
            text_content = df.to_string(index=False)
            
            # Create table structure
            tables = [{
                'data': df.values.tolist(),
                'columns': df.columns.tolist(),
                'rows': len(df),
                'cols': len(df.columns)
            }]
            
            # Basic metadata
            metadata = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.to_dict()
            }
            
            return ExtractedContent(
                text=text_content,
                metadata=metadata,
                tables=tables,
                images=[],
                structure={'type': 'csv', 'shape': df.shape}
            )
        
        except Exception as e:
            return ExtractedContent(
                text="",
                metadata={'filename': filename, 'error': True},
                tables=[],
                images=[],
                structure={},
                errors=[f"CSV extraction failed: {str(e)}"]
            )