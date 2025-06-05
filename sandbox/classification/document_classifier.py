"""
Document classifier that leverages existing FinDocPipeline capabilities.
Provides lightweight classification for multi-document processing.
"""

import streamlit as st
import re
from typing import Dict, Any, Optional
import tempfile
import os

class DocumentClassifier:
    """
    Lightweight document classifier that works with existing FinDocPipeline components.
    Uses rule-based classification optimized for financial documents.
    """
    
    def __init__(self):
        self.classification_patterns = {
            'transcript': {
                'filename_indicators': ['transcript', 'call', 'earnings', 'conference'],
                'content_indicators': [
                    r'operator[:\s]',
                    r'analyst[:\s]',
                    r'q&a|question.{0,10}answer',
                    r'earnings call|conference call',
                    r'thank you.{0,20}operator',
                    r'next question',
                    r'[A-Z][a-z]+ [A-Z][a-z]+:',  # Speaker patterns
                ],
                'weight': 2.0
            },
            'presentation': {
                'filename_indicators': ['presentation', 'slides', 'deck', 'ppt'],
                'content_indicators': [
                    r'slide \d+|next slide',
                    r'agenda|overview',
                    r'key highlights|financial highlights',
                    r'moving to slide|turn to slide',
                    r'as you can see on the slide'
                ],
                'weight': 1.8
            },
            'financial_summary': {
                'filename_indicators': ['summary', 'financial', 'statement', 'report'],
                'content_indicators': [
                    r'income statement|profit.{0,10}loss',
                    r'balance sheet',
                    r'cash flow statement',
                    r'financial statements?',
                    r'quarterly results?',
                    r'revenue.*\$|net income.*\$'
                ],
                'weight': 1.5
            }
        }
    
    def classify_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Classify an uploaded file using filename and content analysis.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dict containing classification results
        """
        
        try:
            # Get filename-based classification
            filename_scores = self._classify_by_filename(uploaded_file.name)
            
            # Get content-based classification (sample first 2000 chars for performance)
            content_sample = self._extract_content_sample(uploaded_file)
            content_scores = self._classify_by_content(content_sample)
            
            # Combine scores
            combined_scores = self._combine_classification_scores(filename_scores, content_scores)
            
            # Determine final classification
            if not combined_scores:
                return {
                    'document_type': 'unknown',
                    'confidence': 0.0,
                    'method': 'rule_based',
                    'scores': {},
                    'reasoning': 'No classification patterns matched'
                }
            
            # Get highest scoring type
            best_type = max(combined_scores, key=combined_scores.get)
            confidence = min(combined_scores[best_type] / 10.0, 1.0)  # Normalize to 0-1
            
            return {
                'document_type': best_type,
                'confidence': confidence,
                'method': 'rule_based',
                'scores': combined_scores,
                'reasoning': f'Classified as {best_type} based on filename and content patterns'
            }
        
        except Exception as e:
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'method': 'error',
                'error': str(e),
                'reasoning': f'Classification failed: {str(e)}'
            }
    
    def _classify_by_filename(self, filename: str) -> Dict[str, float]:
        """Classify document based on filename patterns"""
        
        scores = {}
        filename_lower = filename.lower()
        
        for doc_type, patterns in self.classification_patterns.items():
            score = 0.0
            
            # Check filename indicators
            for indicator in patterns['filename_indicators']:
                if indicator in filename_lower:
                    score += patterns['weight'] * 2  # Filename gets higher weight
            
            if score > 0:
                scores[doc_type] = score
        
        return scores
    
    def _extract_content_sample(self, uploaded_file) -> str:
        """Extract a sample of content for classification"""
        
        try:
            # Save current position
            current_pos = uploaded_file.tell()
            
            # Read content based on file type
            if uploaded_file.name.lower().endswith('.pdf'):
                content_sample = self._extract_pdf_sample(uploaded_file)
            elif uploaded_file.name.lower().endswith('.docx'):
                content_sample = self._extract_docx_sample(uploaded_file)
            else:
                # For other text files, read directly
                uploaded_file.seek(0)
                content_bytes = uploaded_file.read(2000)  # First 2KB
                content_sample = content_bytes.decode('utf-8', errors='ignore')
            
            # Restore file position
            uploaded_file.seek(current_pos)
            
            return content_sample
        
        except Exception as e:
            # Restore file position on error
            try:
                uploaded_file.seek(current_pos)
            except:
                pass
            return ""
    
    def _extract_pdf_sample(self, uploaded_file) -> str:
        """Extract sample text from PDF for classification"""
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Try to extract text using available methods
            content_sample = ""
            
            # Try pdfplumber first
            try:
                import pdfplumber
                with pdfplumber.open(tmp_file_path) as pdf:
                    # Get text from first page only for classification
                    if len(pdf.pages) > 0:
                        page_text = pdf.pages[0].extract_text()
                        if page_text:
                            content_sample = page_text[:2000]  # First 2000 chars
            except ImportError:
                pass
            
            # Fallback to PyMuPDF if pdfplumber not available
            if not content_sample:
                try:
                    import fitz
                    doc = fitz.open(tmp_file_path)
                    if len(doc) > 0:
                        page = doc.load_page(0)
                        content_sample = page.get_text()[:2000]
                    doc.close()
                except ImportError:
                    pass
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            return content_sample
        
        except Exception:
            return ""
    
    def _extract_docx_sample(self, uploaded_file) -> str:
        """Extract sample text from DOCX for classification"""
        
        try:
            from docx import Document
            import io
            
            # Read file content
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            # Create document from bytes
            doc = Document(io.BytesIO(file_content))
            
            # Extract text from first few paragraphs
            text_parts = []
            char_count = 0
            
            for paragraph in doc.paragraphs:
                if char_count >= 2000:  # Limit to 2000 chars
                    break
                
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
                    char_count += len(paragraph.text)
            
            return '\n'.join(text_parts)
        
        except Exception:
            return ""
    
    def _classify_by_content(self, content: str) -> Dict[str, float]:
        """Classify document based on content patterns"""
        
        scores = {}
        
        if not content:
            return scores
        
        content_lower = content.lower()
        
        for doc_type, patterns in self.classification_patterns.items():
            score = 0.0
            
            # Check content indicators
            for pattern in patterns['content_indicators']:
                matches = len(re.findall(pattern, content_lower))
                if matches > 0:
                    score += matches * patterns['weight']
            
            if score > 0:
                scores[doc_type] = score
        
        return scores
    
    def _combine_classification_scores(self, filename_scores: Dict[str, float], content_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine filename and content classification scores"""
        
        combined_scores = {}
        
        # Get all document types that had any score
        all_types = set(filename_scores.keys()) | set(content_scores.keys())
        
        for doc_type in all_types:
            filename_score = filename_scores.get(doc_type, 0.0)
            content_score = content_scores.get(doc_type, 0.0)
            
            # Weighted combination (filename gets 40%, content gets 60%)
            combined_score = (filename_score * 0.4) + (content_score * 0.6)
            
            if combined_score > 0:
                combined_scores[doc_type] = combined_score
        
        return combined_scores
    
    @st.cache_data
    def get_classification_stats(_self) -> Dict[str, Any]:
        """Get classification statistics and patterns"""
        
        stats = {
            'supported_types': list(_self.classification_patterns.keys()),
            'total_patterns': sum(
                len(patterns['filename_indicators']) + len(patterns['content_indicators'])
                for patterns in _self.classification_patterns.values()
            ),
            'pattern_details': {}
        }
        
        for doc_type, patterns in _self.classification_patterns.items():
            stats['pattern_details'][doc_type] = {
                'filename_patterns': len(patterns['filename_indicators']),
                'content_patterns': len(patterns['content_indicators']),
                'weight': patterns['weight']
            }
        
        return stats