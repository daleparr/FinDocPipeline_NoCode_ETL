import streamlit as st
import os

# Configure Streamlit to reduce console warnings
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
import pandas as pd
import tempfile
from datetime import datetime
import os
import re
import subprocess
import sys
import json
import numpy as np
from PIL import Image
import io
import base64

# Try to import optional computer vision libraries
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Import the comprehensive parser from previous implementation
class ComprehensiveFinancialParser:
    """Comprehensive parser that captures ALL text plus enhanced table/chart interpretation"""
    
    def __init__(self):
        self.pdf_methods = []
        self._check_available_methods()
    
    def _check_available_methods(self):
        """Check which PDF processing methods are available"""
        try:
            import pdfplumber
            self.pdf_methods.append('pdfplumber')
        except ImportError:
            pass
        
        try:
            import fitz
            self.pdf_methods.append('pymupdf')
        except ImportError:
            pass
    
    def extract_comprehensive_data(self, pdf_path):
        """Extract ALL text content plus enhanced structural analysis"""
        if 'pdfplumber' in self.pdf_methods:
            return self._extract_with_pdfplumber(pdf_path)
        elif 'pymupdf' in self.pdf_methods:
            return self._extract_with_pymupdf(pdf_path)
        else:
            raise Exception("No PDF processing libraries available")
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Comprehensive extraction using pdfplumber"""
        import pdfplumber
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract ALL text content
                full_text = page.extract_text() or ""
                
                page_data = {
                    'page': page_num + 1,
                    'method': 'pdfplumber_comprehensive',
                    'full_text': full_text,
                    'word_count': len(full_text.split()),
                    'char_count': len(full_text),
                    'line_count': len(full_text.split('\n')),
                    'tables': [],
                    'financial_metrics': {},
                    'chart_indicators': []
                }
                
                # Extract tables with full structure
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 0:
                                table_text = self._table_to_text(table)
                                page_data['tables'].append({
                                    'table_id': table_idx,
                                    'table_text': table_text,
                                    'row_count': len(table),
                                    'col_count': len(table[0]) if table else 0
                                })
                except Exception as e:
                    st.warning(f"Table extraction error on page {page_num + 1}: {str(e)}")
                
                # Enhanced financial analysis
                page_data['financial_metrics'] = self._extract_financial_data(full_text)
                page_data['chart_indicators'] = self._detect_chart_elements(full_text)
                
                pages_data.append(page_data)
        
        return pages_data
    
    def _extract_with_pymupdf(self, pdf_path):
        """Comprehensive extraction using PyMuPDF"""
        import fitz
        pages_data = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text = page.get_text()
            
            page_data = {
                'page': page_num + 1,
                'method': 'pymupdf_comprehensive',
                'full_text': full_text,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'line_count': len(full_text.split('\n')),
                'tables': [],
                'financial_metrics': self._extract_financial_data(full_text),
                'chart_indicators': self._detect_chart_elements(full_text)
            }
            
            pages_data.append(page_data)
        
        doc.close()
        return pages_data
    
    def _table_to_text(self, table):
        """Convert table structure to readable text"""
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            if row:
                clean_row = [str(cell) if cell is not None else "" for cell in row]
                text_lines.append(" | ".join(clean_row))
        
        return "\n".join(text_lines)
    
    def _extract_financial_data(self, text):
        """Extract financial metrics and data points"""
        metrics = {}
        
        # Revenue patterns
        revenue_patterns = [
            r'revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?',
            r'total revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?',
            r'net revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?'
        ]
        
        for pattern in revenue_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metrics['revenue'] = match.group(1)
                break
        
        # Profit patterns
        profit_patterns = [
            r'net income[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?',
            r'profit[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?',
            r'earnings[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?'
        ]
        
        for pattern in profit_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                metrics['profit'] = match.group(1)
                break
        
        return metrics
    
    def _detect_chart_elements(self, text):
        """Detect chart and visualization indicators"""
        indicators = []
        
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        chart_keywords = ['chart', 'graph', 'figure', 'exhibit', 'table', 'diagram']
        for keyword in chart_keywords:
            if keyword.lower() in text.lower():
                indicators.append(keyword)
        
        return indicators

class NLPDataProcessor:
    """Process raw data for NLP readiness with comprehensive cleaning"""
    
    def create_raw_csv(self, pages_data):
        """Create raw CSV from extracted pages data"""
        rows = []
        
        for page_data in pages_data:
            # Main text content
            if page_data.get('full_text'):
                rows.append({
                    'page_number': page_data['page'],
                    'content_type': 'main_text',
                    'raw_text': page_data['full_text'],
                    'word_count': page_data.get('word_count', 0),
                    'char_count': page_data.get('char_count', 0),
                    'extraction_method': page_data.get('method', 'unknown')
                })
            
            # Table content
            for table in page_data.get('tables', []):
                rows.append({
                    'page_number': page_data['page'],
                    'content_type': 'table',
                    'raw_text': table.get('table_text', ''),
                    'word_count': len(table.get('table_text', '').split()),
                    'char_count': len(table.get('table_text', '')),
                    'extraction_method': page_data.get('method', 'unknown')
                })
        
        return pd.DataFrame(rows)
    
    def clean_for_nlp(self, raw_df):
        """Clean and prepare data for NLP processing"""
        nlp_df = raw_df.copy()
        
        # Text cleaning
        nlp_df['cleaned_text'] = nlp_df['raw_text'].apply(self._clean_text)
        nlp_df['sentence_count'] = nlp_df['cleaned_text'].apply(self._count_sentences)
        nlp_df['has_financial_terms'] = nlp_df['cleaned_text'].apply(self._has_financial_terms)
        
        # Filter out very short or empty content
        nlp_df = nlp_df[nlp_df['cleaned_text'].str.len() > 10]
        
        return nlp_df, raw_df
    
    def _clean_text(self, text):
        """Clean text for NLP processing"""
        if pd.isna(text):
            return ""
        
        # Remove extra whitespace but preserve line breaks initially
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Keep financial symbols and basic punctuation
        # Allow: letters, numbers, spaces, basic punctuation, currency symbols, mathematical symbols
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\%\$£€¥\+\=\|\n\r]', ' ', text)
        
        # Convert line breaks to spaces
        text = re.sub(r'[\n\r]+', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _count_sentences(self, text):
        """Count sentences in text"""
        if not text:
            return 0
        return len(re.split(r'[.!?]+', text))
    
    def _has_financial_terms(self, text):
        """Check if text contains financial terms"""
        financial_terms = [
            'revenue', 'profit', 'earnings', 'income', 'assets', 'liabilities',
            'equity', 'cash flow', 'margin', 'growth', 'return', 'investment'
        ]
        
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text_lower = text.lower()
        return any(term in text_lower for term in financial_terms)

class NLPDatasetExporter:
    """Create specialized NLP datasets for training and analysis"""
    
    def __init__(self):
        self.financial_labels = [
            'revenue', 'profit', 'loss', 'assets', 'liabilities', 'equity',
            'cash_flow', 'earnings', 'income', 'expenses', 'margin', 'growth',
            'investment', 'debt', 'return', 'dividend', 'share', 'stock'
        ]
    
    def create_nlp_dataset(self, nlp_df, metrics_df=None):
        """Create comprehensive NLP dataset with labels and features"""
        nlp_dataset = []
        
        for idx, row in nlp_df.iterrows():
            # Basic text features
            text = row['cleaned_text']
            
            # Create dataset entry
            entry = {
                'id': f"doc_{row['page_number']}_{idx}",
                'page_number': row['page_number'],
                'content_type': row['content_type'],
                'text': text,
                'word_count': row['word_count'],
                'char_count': row['char_count'],
                'sentence_count': row['sentence_count'],
                'has_financial_terms': row['has_financial_terms'],
                'extraction_method': row['extraction_method']
            }
            
            # Add text classification labels
            entry.update(self._create_text_labels(text))
            
            # Add readability metrics
            entry.update(self._calculate_readability_metrics(text))
            
            # Add financial entity indicators
            entry.update(self._extract_financial_entities(text))
            
            # Add contextual features
            entry.update(self._extract_contextual_features(text))
            
            nlp_dataset.append(entry)
        
        return pd.DataFrame(nlp_dataset)
    
    def _create_text_labels(self, text):
        """Create classification labels for text"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text_lower = text.lower()
        
        labels = {
            'is_financial_statement': any(term in text_lower for term in
                ['balance sheet', 'income statement', 'cash flow statement', 'statement of']),
            'is_narrative_text': len(text.split('.')) > 3 and not ('|' in text or '\t' in text),
            'is_tabular_data': '|' in text or '\t' in text or text.count('\n') > text.count('.'),
            'contains_numbers': bool(re.search(r'\d+', text)),
            'contains_percentages': '%' in text,
            'contains_currency': bool(re.search(r'[\$£€¥]', text)),
            'is_executive_summary': any(term in text_lower for term in
                ['executive summary', 'overview', 'highlights', 'key points']),
            'is_risk_disclosure': any(term in text_lower for term in
                ['risk', 'uncertainty', 'forward-looking', 'may', 'could', 'might']),
            'is_performance_metric': any(term in text_lower for term in
                ['performance', 'results', 'achievement', 'target', 'goal'])
        }
        
        return labels
    
    def _calculate_readability_metrics(self, text):
        """Calculate basic readability metrics"""
        if not text or len(text.strip()) == 0:
            return {
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'complexity_score': 0
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Simple complexity score based on word and sentence length
        complexity_score = (avg_word_length * 0.4) + (avg_sentence_length * 0.6)
        
        return {
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'complexity_score': round(complexity_score, 2)
        }
    
    def _extract_financial_entities(self, text):
        """Extract and count financial entities"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text_lower = text.lower()
        
        entity_counts = {}
        for label in self.financial_labels:
            count = len(re.findall(r'\b' + re.escape(label) + r'\b', text_lower))
            entity_counts[f'{label}_count'] = count
        
        # Add total financial entity density
        total_financial_terms = sum(entity_counts.values())
        word_count = len(text.split())
        financial_density = total_financial_terms / word_count if word_count > 0 else 0
        
        entity_counts['financial_entity_density'] = round(financial_density, 4)
        
        return entity_counts
    
    def _extract_contextual_features(self, text):
        """Extract contextual features for NLP"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text_lower = text.lower()
        
        features = {
            'has_time_references': bool(re.search(r'\b(year|month|quarter|q[1-4]|\d{4})\b', text_lower)),
            'has_comparison_terms': any(term in text_lower for term in
                ['compared to', 'versus', 'vs', 'increase', 'decrease', 'higher', 'lower']),
            'has_future_tense': any(term in text_lower for term in
                ['will', 'expect', 'forecast', 'project', 'anticipate', 'plan']),
            'has_past_tense': any(term in text_lower for term in
                ['was', 'were', 'had', 'achieved', 'reported', 'recorded']),
            'sentiment_indicators': self._basic_sentiment_analysis(text_lower),
            'formality_score': self._calculate_formality_score(text)
        }
        
        return features
    
    def _basic_sentiment_analysis(self, text_lower):
        """Basic sentiment analysis for financial text"""
        positive_terms = ['growth', 'increase', 'profit', 'success', 'strong', 'improved', 'gain']
        negative_terms = ['loss', 'decrease', 'decline', 'weak', 'poor', 'risk', 'challenge']
        
        positive_count = sum(1 for term in positive_terms if term in text_lower)
        negative_count = sum(1 for term in negative_terms if term in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_formality_score(self, text):
        """Calculate formality score based on text characteristics"""
        if not text:
            return 0
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Indicators of formal text
        formal_indicators = 0
        
        # Long sentences
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length > 15:
            formal_indicators += 1
        
        # Complex words (more than 6 characters)
        words = text.split()
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_ratio = complex_words / len(words) if words else 0
        if complex_ratio > 0.3:
            formal_indicators += 1
        
        # Passive voice indicators
        if any(term in text.lower() for term in ['is', 'was', 'were', 'been', 'being']):
            formal_indicators += 1
        
        return formal_indicators / 3  # Normalize to 0-1 scale
    
    def export_nlp_csv(self, nlp_dataset_df):
        """Export NLP dataset as CSV"""
        return nlp_dataset_df.to_csv(index=False)
    
    def export_nlp_json(self, nlp_dataset_df):
        """Export NLP dataset as JSON with proper structure"""
        # Create a structured JSON format suitable for NLP frameworks
        nlp_json = {
            "dataset_info": {
                "name": "Financial Document NLP Dataset",
                "version": "1.0",
                "description": "Processed financial document text with NLP features and labels",
                "created_at": datetime.now().isoformat(),
                "total_samples": len(nlp_dataset_df)
            },
            "features": {
                "text_features": ["text", "word_count", "char_count", "sentence_count"],
                "classification_labels": [
                    "is_financial_statement", "is_narrative_text", "is_tabular_data",
                    "is_executive_summary", "is_risk_disclosure", "is_performance_metric"
                ],
                "readability_metrics": ["avg_word_length", "avg_sentence_length", "complexity_score"],
                "entity_counts": [f"{label}_count" for label in self.financial_labels],
                "contextual_features": [
                    "has_time_references", "has_comparison_terms", "has_future_tense",
                    "has_past_tense", "sentiment_indicators", "formality_score"
                ]
            },
            "data": nlp_dataset_df.to_dict(orient="records")
        }
        
        return json.dumps(nlp_json, indent=2, ensure_ascii=False)

class EnhancedVisualParser:
    """Enhanced parser with computer vision and OCR capabilities for charts, graphs, and images"""
    
    def __init__(self):
        self.ocr_available = False
        self.cv_available = False
        self._check_cv_capabilities()
        
        # Enhanced financial dictionaries for cleaning and sorting
        self.financial_terms_dict = {
            'revenue_terms': ['revenue', 'sales', 'income', 'turnover', 'receipts'],
            'profit_terms': ['profit', 'earnings', 'net income', 'ebitda', 'operating income'],
            'asset_terms': ['assets', 'property', 'equipment', 'inventory', 'cash', 'investments'],
            'liability_terms': ['liabilities', 'debt', 'payables', 'obligations', 'borrowings'],
            'equity_terms': ['equity', 'shareholders equity', 'retained earnings', 'capital'],
            'ratio_terms': ['ratio', 'margin', 'return', 'yield', 'percentage', 'rate'],
            'trend_terms': ['increase', 'decrease', 'growth', 'decline', 'improvement', 'deterioration'],
            'time_terms': ['year', 'quarter', 'month', 'annual', 'quarterly', 'monthly', 'ytd', 'q1', 'q2', 'q3', 'q4']
        }
        
        # Chart type indicators
        self.chart_indicators = {
            'bar_chart': ['bar chart', 'bar graph', 'column chart', 'histogram'],
            'line_chart': ['line chart', 'line graph', 'trend line', 'time series'],
            'pie_chart': ['pie chart', 'pie graph', 'donut chart', 'circular chart'],
            'scatter_plot': ['scatter plot', 'scatter chart', 'correlation plot'],
            'table': ['table', 'matrix', 'grid', 'tabular data']
        }
    
    def _check_cv_capabilities(self):
        """Check available computer vision and OCR libraries"""
        self.ocr_available = OCR_AVAILABLE
        self.cv_available = CV_AVAILABLE
        
        if self.ocr_available:
            st.info("✅ OCR capabilities available (Tesseract)")
        else:
            st.warning("⚠️ OCR not available - install pytesseract for enhanced text extraction")
        
        if self.cv_available:
            st.info("✅ Computer Vision capabilities available (OpenCV)")
        else:
            st.warning("⚠️ Computer Vision not available - install opencv-python for image analysis")
    
    def extract_visual_data(self, pdf_path):
        """Extract visual elements including charts, graphs, and images with OCR"""
        visual_data = []
        
        if not PYMUPDF_AVAILABLE:
            st.warning("⚠️ PyMuPDF not available - visual extraction limited")
            return visual_data
        
        try:
            import fitz  # PyMuPDF for image extraction
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_visual_data = {
                    'page': page_num + 1,
                    'images': [],
                    'charts_detected': [],
                    'ocr_text': '',
                    'visual_metrics': {}
                }
                
                # Extract images from page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Analyze image
                            image_analysis = self._analyze_image(img_pil, page_num + 1, img_index)
                            page_visual_data['images'].append(image_analysis)
                            
                            # Perform OCR if available
                            if self.ocr_available:
                                ocr_text = self._extract_text_from_image(img_pil)
                                page_visual_data['ocr_text'] += f" {ocr_text}"
                        
                        pix = None
                    except Exception as e:
                        st.warning(f"Error processing image {img_index} on page {page_num + 1}: {str(e)}")
                
                # Detect chart types and extract data
                page_visual_data['charts_detected'] = self._detect_chart_types(page_visual_data['ocr_text'])
                page_visual_data['visual_metrics'] = self._extract_visual_metrics(page_visual_data['ocr_text'])
                
                visual_data.append(page_visual_data)
            
            doc.close()
            
        except Exception as e:
            st.error(f"Error in visual data extraction: {str(e)}")
        
        return visual_data
    
    def _analyze_image(self, img_pil, page_num, img_index):
        """Analyze individual image for chart/graph characteristics"""
        analysis = {
            'image_id': f"page_{page_num}_img_{img_index}",
            'size': img_pil.size,
            'mode': img_pil.mode,
            'is_chart': False,
            'chart_type': None,
            'contains_text': False,
            'color_analysis': {}
        }
        
        try:
            # Convert to numpy array for analysis
            img_array = np.array(img_pil)
            
            # Basic image analysis
            analysis['color_analysis'] = {
                'mean_brightness': np.mean(img_array),
                'has_multiple_colors': len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)) > 10,
                'aspect_ratio': img_pil.size[0] / img_pil.size[1]
            }
            
            # Computer vision analysis if available
            if self.cv_available:
                analysis.update(self._cv_chart_detection(img_array))
            
        except Exception as e:
            st.warning(f"Error in image analysis: {str(e)}")
        
        return analysis
    
    def _cv_chart_detection(self, img_array):
        """Use computer vision to detect chart elements"""
        if not CV_AVAILABLE:
            return {'cv_available': False}
        
        try:
            import cv2
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect lines (potential chart axes)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            # Detect contours (potential chart elements)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return {
                'lines_detected': len(lines) if lines is not None else 0,
                'contours_detected': len(contours),
                'likely_chart': (len(lines) if lines is not None else 0) > 2 and len(contours) > 5,
                'chart_complexity': len(contours) / 10 if contours else 0
            }
            
        except Exception as e:
            return {'cv_error': str(e)}
    
    def _extract_text_from_image(self, img_pil):
        """Extract text from image using OCR"""
        if not OCR_AVAILABLE:
            return "OCR not available"
        
        try:
            import pytesseract
            
            # Configure OCR for better financial document recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()%-$£€¥'
            
            text = pytesseract.image_to_string(img_pil, config=custom_config)
            return text.strip()
            
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def _detect_chart_types(self, text):
        """Detect chart types from extracted text"""
        detected_charts = []
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text_lower = text.lower()
        
        for chart_type, indicators in self.chart_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    detected_charts.append({
                        'type': chart_type,
                        'indicator': indicator,
                        'confidence': 0.8 if 'chart' in indicator else 0.6
                    })
        
        return detected_charts
    
    def _extract_visual_metrics(self, ocr_text):
        """Extract financial metrics from OCR text using enhanced dictionaries"""
        metrics = {}
        
        # Clean and normalize OCR text
        cleaned_text = self._clean_ocr_text(ocr_text)
        
        # Extract metrics using financial dictionaries
        for category, terms in self.financial_terms_dict.items():
            category_metrics = []
            for term in terms:
                # Look for numerical values near financial terms
                pattern = rf'{re.escape(term)}\s*[:\-]?\s*([\d,\.]+)\s*([%$£€¥]?)\s*(million|billion|m|b|k)?'
                matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
                
                for match in matches:
                    category_metrics.append({
                        'term': term,
                        'value': match.group(1),
                        'currency': match.group(2),
                        'unit': match.group(3),
                        'context': cleaned_text[max(0, match.start()-30):match.end()+30]
                    })
            
            if category_metrics:
                metrics[category] = category_metrics
        
        return metrics
    
    def _clean_ocr_text(self, text):
        """Clean and normalize OCR text for better extraction"""
        if not text:
            return ""
        
        # Fix common OCR errors
        text = re.sub(r'[|l1]', 'I', text)  # Fix common character misreads
        text = re.sub(r'[O0]', '0', text)   # Normalize zeros
        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\%\$£€¥]', ' ', text)  # Remove noise
        
        return text.strip()
    
    def create_enhanced_dataset(self, visual_data, existing_nlp_df):
        """Create enhanced dataset combining visual and text data"""
        enhanced_rows = []
        
        for page_data in visual_data:
            page_num = page_data['page']
            
            # Add visual content as separate entries
            if page_data['ocr_text']:
                enhanced_rows.append({
                    'page_number': page_num,
                    'content_type': 'visual_ocr',
                    'raw_text': page_data['ocr_text'],
                    'word_count': len(page_data['ocr_text'].split()),
                    'char_count': len(page_data['ocr_text']),
                    'extraction_method': 'ocr_visual',
                    'images_count': len(page_data['images']),
                    'charts_detected': len(page_data['charts_detected']),
                    'visual_metrics_count': len(page_data['visual_metrics'])
                })
            
            # Add chart-specific entries
            for chart in page_data['charts_detected']:
                enhanced_rows.append({
                    'page_number': page_num,
                    'content_type': f"chart_{chart['type']}",
                    'raw_text': f"Chart detected: {chart['type']} - {chart['indicator']}",
                    'word_count': 5,
                    'char_count': 50,
                    'extraction_method': 'computer_vision',
                    'chart_confidence': chart['confidence'],
                    'chart_type': chart['type']
                })
        
        # Combine with existing NLP data
        if enhanced_rows:
            enhanced_df = pd.concat([
                existing_nlp_df,
                pd.DataFrame(enhanced_rows)
            ], ignore_index=True)
        else:
            # If no visual data, return the original NLP data
            enhanced_df = existing_nlp_df.copy()
        
        return enhanced_df

class DeduplicatedMetricsExtractor:
    """Extract financial metrics with deduplication logic"""
    
    def __init__(self):
        self.metric_patterns = {
            'revenue': [
                r'(?:total\s+)?revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
                r'(?:net\s+)?sales[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'net_income': [
                r'net\s+income[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
                r'net\s+profit[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'total_assets': [
                r'total\s+assets[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'shareholders_equity': [
                r'(?:shareholders?\s+|stockholders?\s+)?equity[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'cash_and_equivalents': [
                r'cash\s+(?:and\s+)?(?:cash\s+)?equivalents[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'operating_income': [
                r'operating\s+income[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'gross_profit': [
                r'gross\s+profit[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'ebitda': [
                r'ebitda[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
            ],
            'eps': [
                r'(?:earnings\s+per\s+share|eps)[:\s]+[\$£€]?([\d,\.]+)'
            ],
            'roe': [
                r'(?:return\s+on\s+equity|roe)[:\s]+([\d,\.]+)%?'
            ],
            'roa': [
                r'(?:return\s+on\s+assets|roa)[:\s]+([\d,\.]+)%?'
            ],
            'debt_to_equity': [
                r'debt[:\s\-]+to[:\s\-]+equity[:\s]+([\d,\.]+)'
            ]
        }
    
    def extract_metrics_enhanced(self, nlp_df):
        """Extract metrics with enhanced deduplication and debugging"""
        all_metrics = []
        debug_info = []
        
        for idx, row in nlp_df.iterrows():
            page_num = row['page_number']
            text = row['cleaned_text']
            
            page_debug = {
                'page_number': page_num,
                'text_length': len(text),
                'matches_found': 0,
                'metrics_extracted': []
            }
            
            for metric_name, patterns in self.metric_patterns.items():
                for pattern in patterns:
                    matches = list(re.finditer(pattern, text, re.IGNORECASE))
                    
                    for match in matches:
                        value = match.group(1)
                        unit = match.group(2) if len(match.groups()) > 1 else None
                        
                        # Calculate confidence based on context
                        confidence = self._calculate_confidence(text, match, metric_name)
                        
                        metric_entry = {
                            'page_number': page_num,
                            'metric_name': metric_name,
                            'metric_value': value,
                            'unit': unit,
                            'confidence': confidence,
                            'context': text[max(0, match.start()-50):match.end()+50],
                            'pattern_used': pattern
                        }
                        
                        all_metrics.append(metric_entry)
                        page_debug['matches_found'] += 1
                        page_debug['metrics_extracted'].append(metric_name)
            
            debug_info.append(page_debug)
        
        metrics_df = pd.DataFrame(all_metrics)
        debug_df = pd.DataFrame(debug_info)
        
        # Apply deduplication
        if len(metrics_df) > 0:
            metrics_df = self._deduplicate_metrics(metrics_df)
        
        return metrics_df, debug_df
    
    def _calculate_confidence(self, text, match, metric_name):
        """Calculate confidence score for extracted metric"""
        confidence = 0.5  # Base confidence
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Check for table context
        if 'table' in text.lower() or '|' in text:
            confidence += 0.2
        
        # Check for financial statement context
        financial_contexts = ['balance sheet', 'income statement', 'cash flow', 'financial']
        if any(ctx in text.lower() for ctx in financial_contexts):
            confidence += 0.2
        
        # Check for proper formatting
        match_text = match.group(0) if match else ""
        if '$' in match_text or '£' in match_text or '€' in match_text:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_metrics(self, metrics_df):
        """Remove duplicate metrics using confidence-based selection"""
        if len(metrics_df) == 0:
            return metrics_df
        
        # Group by page and metric name, keep highest confidence
        deduplicated = metrics_df.loc[
            metrics_df.groupby(['page_number', 'metric_name'])['confidence'].idxmax()
        ]
        
        return deduplicated.reset_index(drop=True)
    
    def create_wide_metrics(self, metrics_long_df):
        """Convert long-form metrics to wide-form"""
        if len(metrics_long_df) == 0:
            return pd.DataFrame()
        
        # Pivot the data
        wide_df = metrics_long_df.pivot_table(
            index='page_number',
            columns='metric_name',
            values='metric_value',
            aggfunc='first'
        ).reset_index()
        
        # Clean column names
        wide_df.columns = [str(col) for col in wide_df.columns]
        
        return wide_df

def install_pdfplumber():
    """Install pdfplumber"""
    try:
        st.info("Installing pdfplumber...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        st.success("Successfully installed pdfplumber")
        return True
    except Exception as e:
        st.error(f"Failed to install pdfplumber: {str(e)}")
        return False

def install_cv_libraries():
    """Install computer vision and OCR libraries"""
    try:
        st.info("Installing computer vision libraries...")
        
        # Install OpenCV
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        st.success("✅ Installed OpenCV")
        
        # Install Tesseract OCR
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
        st.success("✅ Installed pytesseract")
        
        # Install additional image processing libraries
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "numpy"])
        st.success("✅ Installed image processing libraries")
        
        st.warning("⚠️ Note: You may need to install Tesseract OCR system binary separately:")
        st.code("# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n# Ubuntu: sudo apt install tesseract-ocr\n# macOS: brew install tesseract")
        
        return True
    except Exception as e:
        st.error(f"Failed to install CV libraries: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="FinDocPipeline - No-Code ETL for Financial Slide Decks",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    
    # Minimal CSS - no meta tags to avoid conflicts
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
        font-style: italic;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Main title with custom styling
    st.markdown('<h1 class="main-title">📊 FinDocPipeline: Your No-Code ETL Solution for Financial Slide Decks</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform messy, unstructured earnings-deck text into clean, analysis-ready data—without requiring any Python</p>', unsafe_allow_html=True)
    
    # Initialize components
    parser = ComprehensiveFinancialParser()
    nlp_processor = NLPDataProcessor()
    metrics_extractor = DeduplicatedMetricsExtractor()
    nlp_exporter = NLPDatasetExporter()
    visual_parser = EnhancedVisualParser()
    
    # Check capabilities
    if not parser.pdf_methods:
        st.warning("⚠️ Processing requires pdfplumber or PyMuPDF!")
        if st.button("🔧 Install pdfplumber"):
            if install_pdfplumber():
                st.rerun()
        return
    else:
        st.success(f"✅ Processing available with: {', '.join(parser.pdf_methods)}")
    
    # Check computer vision capabilities
    if not visual_parser.ocr_available or not visual_parser.cv_available:
        st.info("🔧 Enhanced visual analysis requires additional libraries")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📦 Install Computer Vision Libraries"):
                if install_cv_libraries():
                    st.rerun()
        with col2:
            st.write("**Enhanced features with CV libraries:**")
            st.write("• OCR text extraction from charts/images")
            st.write("• Chart type detection and analysis")
            st.write("• Visual trend pattern recognition")
    
    # File upload
    st.header("📁 Upload Financial Slide Deck")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload earnings presentations, financial slide decks, or investor presentations"
    )
    
    if uploaded_file is not None:
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("📊 EXTRACT: Reading slide deck content..."):
                    pages_data = parser.extract_comprehensive_data(tmp_path)
                
                with st.spinner("🔄 TRANSFORM: Creating raw structured data..."):
                    raw_df = nlp_processor.create_raw_csv(pages_data)
                
                with st.spinner("🧹 TRANSFORM: Cleaning text and normalizing tokens..."):
                    nlp_df, full_df = nlp_processor.clean_for_nlp(raw_df)
                
                with st.spinner("🎯 TRANSFORM: Applying METRIC_PATTERNS regex library..."):
                    metrics_long_df, debug_df = metrics_extractor.extract_metrics_enhanced(nlp_df)
                    metrics_wide_df = metrics_extractor.create_wide_metrics(metrics_long_df)
                
                with st.spinner("👁️ TRANSFORM: Processing visual content with OCR..."):
                    visual_data = visual_parser.extract_visual_data(tmp_path)
                    enhanced_nlp_df = visual_parser.create_enhanced_dataset(visual_data, nlp_df)
                
                with st.spinner("📋 LOAD: Generating clean datasets for analysis..."):
                    nlp_dataset_df = nlp_exporter.create_nlp_dataset(enhanced_nlp_df, metrics_long_df)
                
                st.success(f"✅ ETL Pipeline Complete: {uploaded_file.name} ({len(pages_data)} slides processed)")
                
                # Display results with balanced layout
                st.header("📊 Your Clean, Analysis-Ready Datasets")
                
                # Summary metrics
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                with col1:
                    st.metric("Slides Processed", len(pages_data))
                with col2:
                    total_words = nlp_df['word_count'].sum() if len(nlp_df) > 0 else 0
                    st.metric("Total Words", f"{total_words:,}")
                with col3:
                    st.metric("Clean Text Rows", len(nlp_df))
                with col4:
                    st.metric("Enhanced Rows", len(enhanced_nlp_df))
                with col5:
                    st.metric("NLP Dataset Rows", len(nlp_dataset_df))
                with col6:
                    st.metric("Extracted Metrics", len(metrics_long_df))
                with col7:
                    total_images = sum(len(page['images']) for page in visual_data)
                    total_tables = sum(len(page_data.get('tables', [])) for page_data in pages_data)
                    st.metric("Tables/Charts", f"{total_tables}/{total_images}")
                
                # Three-column balanced layout with export buttons
                st.subheader("📄 Complete Data Preview")
                
                col1, col2, col3 = st.columns(3)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_filename = uploaded_file.name.replace('.pdf', '')
                
                with col1:
                    st.write("**🔤 Raw Data**")
                    st.write("*Original extracted content*")
                    if len(raw_df) > 0:
                        st.dataframe(raw_df.head(10), use_container_width=True, height=400)
                        
                        # Export button for Raw Data
                        raw_csv = raw_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Raw Data CSV",
                            data=raw_csv,
                            file_name=f"raw_data_{base_filename}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("No raw data available")
                
                with col2:
                    st.write("**🤖 NLP-Ready Data**")
                    st.write("*Cleaned and normalized*")
                    if len(nlp_df) > 0:
                        st.dataframe(nlp_df.head(10), use_container_width=True, height=400)
                        
                        # Export button for NLP Data
                        nlp_csv = nlp_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download NLP-Ready CSV",
                            data=nlp_csv,
                            file_name=f"nlp_ready_{base_filename}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("No NLP data available")
                
                with col3:
                    st.write("**📊 Structured Metrics**")
                    st.write("*Extracted financial metrics*")
                    if len(metrics_long_df) > 0:
                        st.dataframe(metrics_long_df.head(10), use_container_width=True, height=400)
                        
                        # Export buttons for Metrics
                        metrics_csv = metrics_long_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Metrics CSV",
                            data=metrics_csv,
                            file_name=f"metrics_long_{base_filename}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        if len(metrics_wide_df) > 0:
                            wide_csv = metrics_wide_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Metrics (Wide) CSV",
                                data=wide_csv,
                                file_name=f"metrics_wide_{base_filename}_{timestamp}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    else:
                        st.info("No metrics extracted")
                
                # Complete Data Preview with all tabs as shown in screenshot
                st.subheader("📄 Complete Data Preview")
                
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Raw Data", "NLP-Ready", "Enhanced NLP", "NLP Dataset", "Visual Analysis", "Metrics (Long)", "Metrics (Wide)", "Debug Info"])
                
                with tab1:
                    st.write("**Raw extracted data with all original content:**")
                    if len(raw_df) > 0:
                        st.dataframe(raw_df, use_container_width=True)
                    else:
                        st.info("No raw data available")
                
                with tab2:
                    st.write("**Cleaned and normalized data ready for NLP processing:**")
                    if len(nlp_df) > 0:
                        st.dataframe(nlp_df, use_container_width=True)
                    else:
                        st.info("No NLP data available")
                
                with tab3:
                    st.write("**Enhanced NLP data with visual content integrated:**")
                    if len(enhanced_nlp_df) > 0:
                        st.dataframe(enhanced_nlp_df, use_container_width=True)
                        
                        # Show content type distribution
                        st.write("**Content Type Distribution:**")
                        content_dist = enhanced_nlp_df['content_type'].value_counts()
                        st.bar_chart(content_dist)
                    else:
                        st.info("No enhanced NLP data available")
                
                with tab4:
                    st.write("**Enhanced NLP dataset with features, labels, and analysis:**")
                    if len(nlp_dataset_df) > 0:
                        st.dataframe(nlp_dataset_df, use_container_width=True)
                        
                        # Show feature summary
                        st.write("**Dataset Features:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**Text Classification Labels:**")
                            classification_cols = [col for col in nlp_dataset_df.columns if col.startswith('is_')]
                            for col in classification_cols[:5]:  # Show first 5
                                true_count = nlp_dataset_df[col].sum()
                                st.write(f"• {col.replace('is_', '').replace('_', ' ').title()}: {true_count}")
                        
                        with col2:
                            st.write("**Readability Metrics:**")
                            if 'avg_word_length' in nlp_dataset_df.columns:
                                avg_word_len = nlp_dataset_df['avg_word_length'].mean()
                                st.write(f"• Avg Word Length: {avg_word_len:.1f}")
                            if 'complexity_score' in nlp_dataset_df.columns:
                                avg_complexity = nlp_dataset_df['complexity_score'].mean()
                                st.write(f"• Avg Complexity: {avg_complexity:.1f}")
                            if 'financial_entity_density' in nlp_dataset_df.columns:
                                avg_density = nlp_dataset_df['financial_entity_density'].mean()
                                st.write(f"• Financial Density: {avg_density:.3f}")
                        
                        with col3:
                            st.write("**Content Distribution:**")
                            if 'sentiment_indicators' in nlp_dataset_df.columns:
                                sentiment_counts = nlp_dataset_df['sentiment_indicators'].value_counts()
                                for sentiment, count in sentiment_counts.items():
                                    st.write(f"• {sentiment.title()}: {count}")
                    else:
                        st.info("No NLP dataset available")
                
                with tab5:
                    st.write("**Visual Analysis: Charts, Graphs, and OCR Results:**")
                    if visual_data:
                        # Visual summary
                        total_images = sum(len(page['images']) for page in visual_data)
                        total_charts = sum(len(page['charts_detected']) for page in visual_data)
                        total_ocr_text = sum(len(page['ocr_text']) for page in visual_data)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Images", total_images)
                        with col2:
                            st.metric("Charts Detected", total_charts)
                        with col3:
                            st.metric("OCR Characters", total_ocr_text)
                        
                        # Page-by-page visual analysis
                        for page_data in visual_data:
                            if page_data['images'] or page_data['charts_detected'] or page_data['ocr_text']:
                                st.write(f"**Page {page_data['page']} Visual Content:**")
                                
                                if page_data['charts_detected']:
                                    st.write("**Charts Detected:**")
                                    for chart in page_data['charts_detected']:
                                        st.write(f"• {chart['type'].title()}: {chart['indicator']} (confidence: {chart['confidence']:.1f})")
                                
                                if page_data['ocr_text']:
                                    with st.expander(f"OCR Text from Page {page_data['page']}"):
                                        st.text(page_data['ocr_text'][:500] + "..." if len(page_data['ocr_text']) > 500 else page_data['ocr_text'])
                                
                                if page_data['visual_metrics']:
                                    with st.expander(f"Visual Metrics from Page {page_data['page']}"):
                                        st.json(page_data['visual_metrics'])
                    else:
                        st.info("No visual data extracted")
                
                with tab6:
                    st.write("**Deduplicated metrics in long form (one row per unique metric):**")
                    if len(metrics_long_df) > 0:
                        st.dataframe(metrics_long_df, use_container_width=True)
                    else:
                        st.info("No metrics extracted from this document")
                
                with tab7:
                    st.write("**Extracted metrics in wide form (one row per page):**")
                    if len(metrics_wide_df) > 0:
                        st.dataframe(metrics_wide_df, use_container_width=True)
                    else:
                        st.info("No wide-form metrics available")
                
                with tab8:
                    st.write("**Extraction debug information:**")
                    if len(debug_df) > 0:
                        st.dataframe(debug_df, use_container_width=True)
                    else:
                        st.info("No debug information available")
                
                # Additional analysis and export options
                st.subheader("📋 Additional Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Processing Stats", "Export Options", "Deduplication Details"])
                
                with tab1:
                    st.write("**Processing statistics:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Content Distribution:**")
                        if len(raw_df) > 0:
                            content_types = raw_df['content_type'].value_counts()
                            st.bar_chart(content_types)
                    
                    with col2:
                        st.write("**Text Length Distribution:**")
                        if len(nlp_df) > 0:
                            st.bar_chart(nlp_df['word_count'].head(20))
                
                with tab2:
                    st.write("**Download processed data:**")
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    base_filename = uploaded_file.name.replace('.pdf', '')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Raw data download
                        if len(raw_df) > 0:
                            raw_csv = raw_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Raw Data CSV",
                                data=raw_csv,
                                file_name=f"raw_data_{base_filename}_{timestamp}.csv",
                                mime="text/csv"
                            )
                        
                        # NLP data download
                        if len(nlp_df) > 0:
                            nlp_csv = nlp_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download NLP-Ready CSV",
                                data=nlp_csv,
                                file_name=f"nlp_ready_{base_filename}_{timestamp}.csv",
                                mime="text/csv"
                            )
                        
                        # Enhanced NLP data download
                        if len(enhanced_nlp_df) > 0:
                            enhanced_csv = enhanced_nlp_df.to_csv(index=False)
                            st.download_button(
                                label="👁️ Download Enhanced NLP CSV",
                                data=enhanced_csv,
                                file_name=f"enhanced_nlp_{base_filename}_{timestamp}.csv",
                                mime="text/csv",
                                help="Includes visual content and OCR data"
                            )
                        
                        # Visual data export
                        if visual_data:
                            visual_json = json.dumps(visual_data, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="👁️ Download Visual Analysis JSON",
                                data=visual_json,
                                file_name=f"visual_analysis_{base_filename}_{timestamp}.json",
                                mime="application/json",
                                help="Complete visual analysis including OCR and chart detection"
                            )
                        
                        # NLP Dataset downloads
                        if len(nlp_dataset_df) > 0:
                            nlp_dataset_csv = nlp_exporter.export_nlp_csv(nlp_dataset_df)
                            st.download_button(
                                label="🤖 Download NLP Dataset CSV",
                                data=nlp_dataset_csv,
                                file_name=f"nlp_dataset_{base_filename}_{timestamp}.csv",
                                mime="text/csv",
                                help="Enhanced NLP dataset with features, labels, and analysis"
                            )
                            
                            nlp_dataset_json = nlp_exporter.export_nlp_json(nlp_dataset_df)
                            st.download_button(
                                label="🤖 Download NLP Dataset JSON",
                                data=nlp_dataset_json,
                                file_name=f"nlp_dataset_{base_filename}_{timestamp}.json",
                                mime="application/json",
                                help="Structured NLP dataset with metadata for ML frameworks"
                            )
                    
                    with col2:
                        # Metrics downloads
                        if len(metrics_long_df) > 0:
                            metrics_csv = metrics_long_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Metrics CSV (Long)",
                                data=metrics_csv,
                                file_name=f"metrics_long_{base_filename}_{timestamp}.csv",
                                mime="text/csv"
                            )
                            
                            if len(metrics_wide_df) > 0:
                                wide_csv = metrics_wide_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Metrics CSV (Wide)",
                                    data=wide_csv,
                                    file_name=f"metrics_wide_{base_filename}_{timestamp}.csv",
                                    mime="text/csv"
                                )
                            
                            # JSON export
                            metrics_json = metrics_long_df.to_json(orient="records", force_ascii=False, indent=2)
                            st.download_button(
                                label="📥 Download Metrics JSON",
                                data=metrics_json,
                                file_name=f"metrics_{base_filename}_{timestamp}.json",
                                mime="application/json"
                            )
                
                with tab3:
                    st.write("**Deduplication process details:**")
                    if len(metrics_long_df) > 0:
                        st.write("**Deduplication Logic:**")
                        st.write("- Groups metrics by (page_number, metric_name)")
                        st.write("- Selects highest confidence score for each group")
                        st.write("- Removes duplicate extractions from same context")
                        
                        st.write("**Confidence Scoring:**")
                        st.write("- Base confidence: 0.5")
                        st.write("- Table context: +0.2")
                        st.write("- Financial statement context: +0.2")
                        st.write("- Currency symbol present: +0.1")
                        
                        if len(metrics_long_df) > 0:
                            confidence_stats = metrics_long_df['confidence'].describe()
                            st.write("**Confidence Statistics:**")
                            st.dataframe(confidence_stats.to_frame().T, use_container_width=True)
                    else:
                        st.info("No deduplication data available")
            
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"Error in balanced processing: {str(e)}")
    
    # Deduplication Summary (moved to bottom)
    if uploaded_file is not None:
        try:
            # Only show if we have processed data
            if 'metrics_long_df' in locals() and 'debug_df' in locals():
                st.markdown("---")
                st.subheader("🔍 Deduplication Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Extraction Summary:**")
                    if len(debug_df) > 0:
                        total_matches = debug_df['matches_found'].sum()
                        pages_with_matches = (debug_df['matches_found'] > 0).sum()
                        avg_text_length = debug_df['text_length'].mean()
                        
                        st.write(f"- Total matches found: {total_matches}")
                        st.write(f"- Pages with matches: {pages_with_matches}/{len(debug_df)}")
                        st.write(f"- Average text length: {avg_text_length:.0f} characters")
                    else:
                        st.info("No debug information available")
                
                with col2:
                    st.write("**Deduplication Results:**")
                    if len(metrics_long_df) > 0:
                        dedup_stats = metrics_long_df.groupby('metric_name').size()
                        unique_metrics = metrics_long_df['metric_name'].nunique()
                        st.write(f"- Unique metrics after deduplication: {len(metrics_long_df)}")
                        st.write(f"- Metric types found: {unique_metrics}")
                        
                        if len(dedup_stats) > 0:
                            st.write("**Metrics by type:**")
                            for metric, count in dedup_stats.head(5).items():
                                st.write(f"  • {metric}: {count}")
                    else:
                        st.info("No metrics extracted")
        except:
            pass
    
    # Footer
    st.markdown("---")
    st.markdown("**FinDocPipeline** - Your No-Code ETL Solution for Financial Slide Decks")

if __name__ == "__main__":
    main()