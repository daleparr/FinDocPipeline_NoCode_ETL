"""
Simple Excel handler for the sandbox environment.
"""

import streamlit as st
import pandas as pd
import io
import time
from typing import Dict, Any, List
from .base_handler import BaseFileHandler, ExtractedContent

class ExcelHandler(BaseFileHandler):
    """Simple Excel file handler"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['xlsx', 'xls']
        self.handler_name = "ExcelHandler"
    
    def validate_file(self, uploaded_file) -> bool:
        """Validate Excel file"""
        try:
            file_ext = self._extract_file_extension(uploaded_file.name)
            if file_ext not in self.supported_extensions:
                return False
            
            if not self._validate_file_size(uploaded_file, max_size_mb=10):
                return False
            
            return True
        except Exception:
            return False
    
    def extract_content(self, uploaded_file) -> ExtractedContent:
        """Extract content from Excel file"""
        start_time = time.time()
        
        try:
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            content = self._extract_excel_cached(file_bytes, uploaded_file.name)
            content.processing_time = time.time() - start_time
            
            return content
        
        except Exception as e:
            return self._create_error_content(
                f"Excel processing error: {str(e)}", 
                uploaded_file.name
            )
    
    @st.cache_data
    def _extract_excel_cached(_self, file_bytes: bytes, filename: str) -> ExtractedContent:
        """Cached Excel extraction"""
        
        try:
            excel_file = io.BytesIO(file_bytes)
            
            # Read all sheets
            excel_data = pd.read_excel(excel_file, sheet_name=None, header=None)
            
            # Convert to text
            text_parts = []
            tables = []
            
            for sheet_name, df in excel_data.items():
                text_parts.append(f"=== SHEET: {sheet_name} ===")
                if not df.empty:
                    text_parts.append(df.to_string(index=False))
                    
                    tables.append({
                        'sheet_name': sheet_name,
                        'data': df.values.tolist(),
                        'columns': df.columns.tolist(),
                        'rows': len(df),
                        'cols': len(df.columns)
                    })
            
            text_content = '\n'.join(text_parts)
            
            # Basic metadata
            metadata = {
                'filename': filename,
                'sheets': list(excel_data.keys()),
                'total_sheets': len(excel_data),
                'handler': 'ExcelHandler'
            }
            
            return ExtractedContent(
                text=text_content,
                metadata=metadata,
                tables=tables,
                images=[],
                structure={'type': 'excel', 'sheets': len(excel_data)}
            )
        
        except Exception as e:
            return ExtractedContent(
                text="",
                metadata={'filename': filename, 'error': True},
                tables=[],
                images=[],
                structure={},
                errors=[f"Excel extraction failed: {str(e)}"]
            )