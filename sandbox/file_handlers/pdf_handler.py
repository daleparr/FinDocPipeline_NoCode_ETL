"""
PDF file handler with Streamlit optimization and caching.
Uses existing FinDocPipeline PDF processing capabilities.
"""

import streamlit as st
import io
import time
from typing import Dict, Any, List, Optional
from .base_handler import BaseFileHandler, ExtractedContent, FileType

# Try to import PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

class PDFHandler(BaseFileHandler):
    """PDF handler that uses existing FinDocPipeline capabilities"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['pdf']
        self.handler_name = "PDFHandler"
        
        # Import existing parser from parent directory
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from FinDocPipeline import ComprehensiveFinancialParser
            self.existing_parser = ComprehensiveFinancialParser()
            self.use_existing_parser = True
        except ImportError:
            self.use_existing_parser = False
            st.warning("⚠️ Could not import existing PDF parser. Using fallback method.")
    
    def validate_file(self, uploaded_file) -> bool:
        """Validate PDF file"""
        try:
            # Check file extension
            if not uploaded_file.name.lower().endswith('.pdf'):
                return False
            
            # Check file size (max 10MB for Streamlit)
            if not self._validate_file_size(uploaded_file, max_size_mb=10):
                return False
            
            # Try to read first few bytes to validate PDF format
            current_pos = uploaded_file.tell()
            file_bytes = uploaded_file.read(1024)
            uploaded_file.seek(current_pos)  # Reset file pointer
            
            return file_bytes.startswith(b'%PDF')
        
        except Exception:
            return False
    
    def extract_content(self, uploaded_file) -> ExtractedContent:
        """Extract comprehensive content from PDF using existing parser"""
        start_time = time.time()
        
        try:
            if self.use_existing_parser:
                return self._extract_with_existing_parser(uploaded_file)
            else:
                return self._extract_with_fallback(uploaded_file)
        
        except Exception as e:
            return self._create_error_content(
                f"PDF processing error: {str(e)}", 
                uploaded_file.name
            )
    
    def _extract_with_existing_parser(self, uploaded_file) -> ExtractedContent:
        """Use existing ComprehensiveFinancialParser"""
        
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Use existing parser
            pages_data = self.existing_parser.extract_comprehensive_data(tmp_file_path)
            
            # Convert to ExtractedContent format
            full_text = []
            tables = []
            images = []
            
            for page_data in pages_data:
                if page_data.get('full_text'):
                    full_text.append(page_data['full_text'])
                
                # Add tables
                for table in page_data.get('tables', []):
                    tables.append({
                        'page': page_data['page'],
                        'data': table.get('table_text', ''),
                        'rows': table.get('row_count', 0),
                        'columns': table.get('col_count', 0)
                    })
            
            # Create metadata
            metadata = {
                'filename': uploaded_file.name,
                'pages': len(pages_data),
                'extraction_method': 'existing_comprehensive_parser',
                'handler': self.handler_name
            }
            
            # Create structure info
            structure = {
                'total_pages': len(pages_data),
                'has_tables': len(tables) > 0,
                'processing_method': 'comprehensive_financial_parser'
            }
            
            return ExtractedContent(
                text='\n'.join(full_text),
                metadata=metadata,
                tables=tables,
                images=images,
                structure=structure,
                raw_data={'pages_data': pages_data}
            )
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def _extract_with_fallback(self, uploaded_file) -> ExtractedContent:
        """Fallback extraction method when existing parser not available"""
        
        if PDFPLUMBER_AVAILABLE:
            return self._extract_with_pdfplumber(uploaded_file)
        elif PYMUPDF_AVAILABLE:
            return self._extract_with_pymupdf(uploaded_file)
        else:
            return ExtractedContent(
                text="PDF processing libraries not available",
                metadata={'filename': uploaded_file.name, 'error': True},
                tables=[],
                images=[],
                structure={},
                errors=["PDF processing libraries (pdfplumber, PyMuPDF) not available"]
            )
    
    def _extract_with_pdfplumber(self, uploaded_file) -> ExtractedContent:
        """Extract using pdfplumber if available"""
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            with pdfplumber.open(tmp_file_path) as pdf:
                text_content = []
                tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 0:
                            tables.append({
                                'page': page_num + 1,
                                'table_index': table_idx + 1,
                                'data': table,
                                'rows': len(table),
                                'columns': len(table[0]) if table else 0
                            })
                
                return ExtractedContent(
                    text='\n'.join(text_content),
                    metadata={
                        'filename': uploaded_file.name,
                        'pages': len(pdf.pages),
                        'extraction_method': 'pdfplumber_fallback'
                    },
                    tables=tables,
                    images=[],
                    structure={'total_pages': len(pdf.pages)}
                )
        
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def _extract_with_pymupdf(self, uploaded_file) -> ExtractedContent:
        """Extract using PyMuPDF if available"""
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            doc = fitz.open(tmp_file_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text_content.append(page_text)
            
            doc.close()
            
            return ExtractedContent(
                text='\n'.join(text_content),
                metadata={
                    'filename': uploaded_file.name,
                    'pages': len(doc),
                    'extraction_method': 'pymupdf_fallback'
                },
                tables=[],
                images=[],
                structure={'total_pages': len(doc)}
            )
        
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def get_processing_estimate(self, uploaded_file) -> Dict[str, Any]:
        """Get processing time estimate for PDF"""
        
        file_size_mb = uploaded_file.size / (1024 * 1024)
        estimated_time = self._estimate_processing_time(uploaded_file.size)
        
        return {
            'estimated_time_seconds': estimated_time,
            'file_size_mb': round(file_size_mb, 2),
            'complexity': 'high' if file_size_mb > 5 else 'medium' if file_size_mb > 2 else 'low',
            'uses_existing_parser': self.use_existing_parser
        }