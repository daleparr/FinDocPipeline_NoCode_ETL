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

class TranscriptParser:
    """Specialized parser for financial transcript documents"""
    
    def __init__(self):
        self.speaker_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)(?:\s*[-–—]\s*([^:]+))?:',  # John Smith - CEO:
            r'([A-Z][A-Z\s]+)(?:\s*[-–—]\s*([^:]+))?:',  # JOHN SMITH - CEO:
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(([^)]+)\):',  # John Smith (CEO):
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[-–—]\s*([^:]+):',  # John Smith - Chief Executive Officer:
        ]
        
        self.role_classifications = {
            'ceo': ['chief executive officer', 'ceo', 'president', 'chairman', 'chief exec'],
            'cfo': ['chief financial officer', 'cfo', 'finance director', 'treasurer'],
            'analyst': ['analyst', 'research analyst', 'equity analyst', 'senior analyst'],
            'investor': ['investor', 'shareholder', 'fund manager', 'portfolio manager'],
            'moderator': ['moderator', 'host', 'operator', 'conference operator'],
            'other': ['director', 'manager', 'officer', 'head of', 'vice president', 'vp']
        }
        
        self.bank_identifiers = [
            'jpmorgan', 'chase', 'bank of america', 'wells fargo', 'citigroup', 'citi',
            'goldman sachs', 'morgan stanley', 'hsbc', 'barclays', 'deutsche bank',
            'credit suisse', 'ubs', 'santander', 'royal bank', 'td bank', 'bmo',
            'scotiabank', 'pnc', 'us bank', 'truist', 'fifth third'
        ]
        
        self.quarter_patterns = [
            r'Q[1-4]\s+20\d{2}',
            r'[Qq]uarter\s+[1-4]\s+20\d{2}',
            r'[Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth\s+[Qq]uarter\s+20\d{2}',
            r'20\d{2}\s+Q[1-4]'
        ]
    
    def extract_transcript_data(self, pdf_path):
        """Extract transcript-specific structured data"""
        try:
            # Use existing parser to get basic text data
            if self._check_pdfplumber():
                pages_data = self._extract_with_pdfplumber(pdf_path)
            elif self._check_pymupdf():
                pages_data = self._extract_with_pymupdf(pdf_path)
            else:
                raise Exception("No PDF processing libraries available")
            
            # Extract transcript-specific information
            transcript_data = []
            bank_name = self._extract_bank_name(pages_data)
            quarter = self._extract_quarter(pages_data)
            
            sequence = 1
            for page_data in pages_data:
                page_num = page_data.get('page', 1)
                full_text = page_data.get('full_text', '')
                
                if full_text:
                    # Split text into speaker segments
                    speaker_segments = self._segment_by_speakers(full_text)
                    
                    for segment in speaker_segments:
                        speaker_name = segment.get('speaker', 'Unknown')
                        role = segment.get('role', 'Unknown')
                        paragraph = segment.get('text', '')
                        word_count = len(paragraph.split()) if paragraph else 0
                        confidence = segment.get('confidence', 0.0)
                        
                        if word_count > 5:  # Only include substantial content
                            transcript_data.append({
                                'bank': bank_name,
                                'quarter': quarter,
                                'speaker': speaker_name,
                                'role': role,
                                'paragraph': paragraph.strip(),
                                'word_count': word_count,
                                'page_number': page_num,
                                'sequence': sequence,
                                'confidence': confidence,
                                'topic_category': self._classify_topic(paragraph)
                            })
                            sequence += 1
            
            return transcript_data
            
        except Exception as e:
            st.error(f"Error in transcript extraction: {str(e)}")
            return []
    
    def _check_pdfplumber(self):
        try:
            import pdfplumber
            return True
        except ImportError:
            return False
    
    def _check_pymupdf(self):
        try:
            import fitz
            return True
        except ImportError:
            return False
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Extract using pdfplumber"""
        import pdfplumber
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                full_text = page.extract_text() or ""
                pages_data.append({
                    'page': page_num + 1,
                    'full_text': full_text,
                    'word_count': len(full_text.split()),
                    'char_count': len(full_text)
                })
        
        return pages_data
    
    def _extract_with_pymupdf(self, pdf_path):
        """Extract using PyMuPDF"""
        import fitz
        pages_data = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text = page.get_text()
            pages_data.append({
                'page': page_num + 1,
                'full_text': full_text,
                'word_count': len(full_text.split()),
                'char_count': len(full_text)
            })
        
        doc.close()
        return pages_data
    
    def _extract_bank_name(self, pages_data):
        """Extract bank name from document content"""
        all_text = ' '.join([page.get('full_text', '') for page in pages_data[:3]])  # Check first 3 pages
        all_text_lower = all_text.lower()
        
        # Check for specific bank patterns with priority order (most specific first)
        bank_patterns = [
            ('citigroup', 'Citigroup'),
            ('citi', 'Citigroup'),
            ('citicorp', 'Citigroup'),
            ('jpmorgan chase', 'JPMorgan Chase'),
            ('jpmorgan', 'JPMorgan Chase'),
            ('bank of america', 'Bank of America'),
            ('wells fargo', 'Wells Fargo'),
            ('goldman sachs', 'Goldman Sachs'),
            ('morgan stanley', 'Morgan Stanley'),
            ('deutsche bank', 'Deutsche Bank'),
            ('credit suisse', 'Credit Suisse'),
            ('royal bank', 'Royal Bank'),
            ('fifth third', 'Fifth Third'),
            ('us bank', 'US Bank'),
            ('td bank', 'TD Bank'),
            ('chase', 'JPMorgan Chase'),  # Put chase after jpmorgan to avoid conflicts
            ('hsbc', 'HSBC'),
            ('barclays', 'Barclays'),
            ('ubs', 'UBS'),
            ('santander', 'Santander'),
            ('bmo', 'BMO'),
            ('scotiabank', 'Scotiabank'),
            ('pnc', 'PNC'),
            ('truist', 'Truist')
        ]
        
        # Check patterns in order of specificity
        for pattern, bank_name in bank_patterns:
            if pattern in all_text_lower:
                return bank_name
        
        return 'Unknown Bank'
    
    def _extract_quarter(self, pages_data):
        """Extract quarter information from document content"""
        all_text = ' '.join([page.get('full_text', '') for page in pages_data[:2]])  # Check first 2 pages
        
        for pattern in self.quarter_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        # Try to extract year at least
        year_match = re.search(r'20\d{2}', all_text)
        if year_match:
            return f"Unknown Quarter {year_match.group()}"
        
        return 'Unknown Quarter'
    
    def _segment_by_speakers(self, text):
        """Segment text by speakers"""
        segments = []
        
        # Try each speaker pattern
        for pattern in self.speaker_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                for i, match in enumerate(matches):
                    speaker_info = match.groups()
                    speaker_name = speaker_info[0].strip()
                    role_info = speaker_info[1] if len(speaker_info) > 1 and speaker_info[1] else ''
                    
                    # Extract text until next speaker or end
                    start_pos = match.end()
                    if i + 1 < len(matches):
                        end_pos = matches[i + 1].start()
                    else:
                        end_pos = len(text)
                    
                    speaker_text = text[start_pos:end_pos].strip()
                    
                    # Classify role
                    role = self._classify_role(role_info + ' ' + speaker_name)
                    
                    if speaker_text and len(speaker_text) > 10:
                        segments.append({
                            'speaker': speaker_name,
                            'role': role,
                            'text': speaker_text,
                            'confidence': 0.8 if role_info else 0.6
                        })
                
                if segments:  # If we found segments with this pattern, use them
                    break
        
        # If no speaker patterns found, treat as single segment
        if not segments and text.strip():
            segments.append({
                'speaker': 'Unknown Speaker',
                'role': 'Unknown',
                'text': text.strip(),
                'confidence': 0.3
            })
        
        return segments
    
    def _classify_role(self, text):
        """Classify speaker role based on text"""
        text_lower = text.lower()
        
        for role, keywords in self.role_classifications.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return role.upper()
        
        return 'OTHER'
    
    def _classify_topic(self, text):
        """Basic topic classification for transcript content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['thank', 'welcome', 'good morning', 'good afternoon']):
            return 'opening_remarks'
        elif any(word in text_lower for word in ['revenue', 'earnings', 'profit', 'income']):
            return 'financial_results'
        elif any(word in text_lower for word in ['outlook', 'guidance', 'expect', 'forecast']):
            return 'guidance'
        elif any(word in text_lower for word in ['question', 'ask', 'answer']):
            return 'qa_session'
        elif any(word in text_lower for word in ['risk', 'challenge', 'concern']):
            return 'risk_discussion'
        else:
            return 'general_discussion'

class TranscriptSchema:
    """Custom schema for transcript documents"""
    
    def __init__(self):
        self.fields = {
            'bank': str,
            'quarter': str,
            'speaker': str,
            'role': str,
            'paragraph': str,
            'word_count': int,
            'page_number': int,
            'sequence': int,
            'confidence': float,
            'topic_category': str
        }
    
    def validate_record(self, record):
        """Validate a single transcript record"""
        errors = []
        
        for field, expected_type in self.fields.items():
            if field not in record:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(record[field], expected_type):
                try:
                    # Try to convert to expected type
                    record[field] = expected_type(record[field])
                except (ValueError, TypeError):
                    errors.append(f"Field {field} should be {expected_type.__name__}")
        
        return len(errors) == 0, errors
    
    def to_dataframe(self, records):
        """Convert transcript records to pandas DataFrame"""
        if not records:
            return pd.DataFrame(columns=list(self.fields.keys()))
        
        # Validate all records
        valid_records = []
        for record in records:
            is_valid, errors = self.validate_record(record)
            if is_valid:
                valid_records.append(record)
            else:
                st.warning(f"Skipping invalid record: {errors}")
        
        return pd.DataFrame(valid_records)
    
    def to_csv(self, records):
        """Convert transcript records to CSV format"""
        df = self.to_dataframe(records)
        return df.to_csv(index=False)
    
    def to_json(self, records, filename="transcript.pdf"):
        """Convert transcript records to structured JSON format"""
        df = self.to_dataframe(records)
        
        if df.empty:
            return json.dumps({"error": "No valid transcript data found"}, indent=2)
        
        # Create metadata
        metadata = {
            "filename": filename,
            "document_type": "transcript",
            "bank": df['bank'].iloc[0] if not df.empty else "Unknown",
            "quarter": df['quarter'].iloc[0] if not df.empty else "Unknown",
            "processed_at": datetime.now().isoformat(),
            "total_records": len(df),
            "total_speakers": df['speaker'].nunique(),
            "total_words": df['word_count'].sum()
        }
        
        # Create speaker summary
        speaker_summary = []
        for speaker in df['speaker'].unique():
            speaker_data = df[df['speaker'] == speaker]
            speaker_summary.append({
                "name": speaker,
                "role": speaker_data['role'].iloc[0],
                "total_words": speaker_data['word_count'].sum(),
                "paragraphs": len(speaker_data),
                "avg_confidence": round(speaker_data['confidence'].mean(), 2)
            })
        
        # Create structured output
        result = {
            "document_metadata": metadata,
            "speakers": speaker_summary,
            "transcript_data": df.to_dict(orient="records")
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)

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
                                    'table_data': table,
                                    'row_count': len(table),
                                    'col_count': len(table[0]) if table else 0
                                })
                except Exception as e:
                    st.warning(f"Table extraction error on page {page_num + 1}: {str(e)}")
                
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
                'financial_metrics': {},
                'chart_indicators': []
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
                # Filter out None values and convert to strings
                clean_row = [str(cell) if cell is not None else "" for cell in row]
                text_lines.append(" | ".join(clean_row))
        
        return "\n".join(text_lines)

def main():
    st.set_page_config(
        page_title="FinDocPipeline v2.0 - Enhanced Document Processing",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎙️ FinDocPipeline v2.0 - Enhanced Document Processing")
    st.markdown("**No-Code ETL Solution for Financial Documents with Transcript Recognition**")
    
    # Document Type Selection
    st.header("📋 Document Type & Upload")
    
    # Document type selection with radio buttons
    document_type = st.radio(
        "Select the type of document you're uploading:",
        options=["📊 Financial Summary", "📈 Presentation", "🎙️ Transcript"],
        index=0,  # Default to financial summary
        help="Choose the document type for optimized processing"
    )
    
    # Dynamic help text based on selection
    if document_type == "🎙️ Transcript":
        st.info("🎙️ **Transcript Mode**: Extracts Bank, Quarter, Speaker, Role, Paragraphs, and Word Counts from earnings call transcripts")
        file_help_text = "Upload earnings call transcripts or investor meeting transcripts"
    else:
        st.info("📊 **Financial Mode**: Extracts metrics, tables, charts, and financial data from presentations and reports")
        file_help_text = "Upload earnings presentations, financial slide decks, or investor presentations"
    
    # File upload with dynamic help text
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help=file_help_text
    )
    
    if uploaded_file is not None:
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Conditional processing based on document type
                if document_type == "🎙️ Transcript":
                    # Transcript processing pipeline
                    st.info("🎙️ Processing as transcript document...")
                    
                    with st.spinner("📊 EXTRACT: Reading transcript content..."):
                        transcript_parser = TranscriptParser()
                        transcript_data = transcript_parser.extract_transcript_data(tmp_path)
                    
                    with st.spinner("🔄 TRANSFORM: Structuring transcript data..."):
                        transcript_schema = TranscriptSchema()
                        transcript_df = transcript_schema.to_dataframe(transcript_data)
                    
                    # Display transcript processing results
                    st.success(f"✅ Transcript Processing Complete: {uploaded_file.name} ({len(transcript_data)} segments processed)")
                    
                    # Show transcript summary
                    if not transcript_df.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Segments", len(transcript_df))
                        with col2:
                            st.metric("Unique Speakers", transcript_df['speaker'].nunique())
                        with col3:
                            st.metric("Total Words", transcript_df['word_count'].sum())
                        with col4:
                            avg_confidence = transcript_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        
                        # Speaker breakdown
                        st.subheader("👥 Speaker Analysis")
                        speaker_stats = transcript_df.groupby(['speaker', 'role']).agg({
                            'word_count': 'sum',
                            'paragraph': 'count',
                            'confidence': 'mean'
                        }).round(2)
                        speaker_stats.columns = ['Total Words', 'Segments', 'Avg Confidence']
                        st.dataframe(speaker_stats, use_container_width=True)
                        
                        # Topic distribution
                        st.subheader("📊 Topic Distribution")
                        topic_stats = transcript_df['topic_category'].value_counts()
                        st.bar_chart(topic_stats)
                        
                        # Export options for transcript
                        st.subheader("📥 Export Transcript Data")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # CSV Export
                            transcript_csv = transcript_schema.to_csv(transcript_data)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            base_filename = uploaded_file.name.replace('.pdf', '')
                            
                            st.download_button(
                                label="📊 Download CSV",
                                data=transcript_csv,
                                file_name=f"{base_filename}_transcript_{timestamp}.csv",
                                mime="text/csv",
                                help="Download transcript data as CSV file"
                            )
                        
                        with col2:
                            # JSON Export
                            transcript_json = transcript_schema.to_json(transcript_data, uploaded_file.name)
                            
                            st.download_button(
                                label="📋 Download JSON",
                                data=transcript_json,
                                file_name=f"{base_filename}_transcript_{timestamp}.json",
                                mime="application/json",
                                help="Download structured transcript data as JSON"
                            )
                        
                        # Show sample data
                        st.subheader("📋 Sample Transcript Data")
                        st.dataframe(transcript_df.head(10), use_container_width=True)
                    
                    else:
                        st.warning("⚠️ No transcript data could be extracted. Please check if the document contains speaker segments.")
                
                else:
                    # Financial document processing pipeline
                    st.info("📊 Processing as financial document...")
                    
                    with st.spinner("📊 EXTRACT: Reading document content..."):
                        parser = ComprehensiveFinancialParser()
                        pages_data = parser.extract_comprehensive_data(tmp_path)
                    
                    with st.spinner("🔄 TRANSFORM: Processing financial data..."):
                        # Create comprehensive dataset
                        all_data = []
                        for page_data in pages_data:
                            page_num = page_data['page']
                            full_text = page_data['full_text']
                            
                            # Basic text processing
                            cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
                            
                            all_data.append({
                                'page_number': page_num,
                                'extraction_method': page_data.get('method', 'unknown'),
                                'full_text': full_text,
                                'cleaned_text': cleaned_text,
                                'word_count': page_data.get('word_count', 0),
                                'char_count': page_data.get('char_count', 0),
                                'line_count': page_data.get('line_count', 0),
                                'table_count': len(page_data.get('tables', [])),
                                'has_tables': len(page_data.get('tables', [])) > 0
                            })
                    
                    # Create DataFrame
                    df = pd.DataFrame(all_data)
                    
                    # Display financial processing results
                    st.success(f"✅ Financial Processing Complete: {uploaded_file.name} ({len(df)} pages processed)")
                    
                    # Show summary metrics
                    if not df.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Pages", len(df))
                        with col2:
                            st.metric("Total Words", df['word_count'].sum())
                        with col3:
                            st.metric("Pages with Tables", df['has_tables'].sum())
                        with col4:
                            st.metric("Total Characters", df['char_count'].sum())
                        
                        # Export options for financial data
                        st.subheader("📥 Export Financial Data")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # CSV Export
                            csv_data = df.to_csv(index=False)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            base_filename = uploaded_file.name.replace('.pdf', '')
                            
                            st.download_button(
                                label="📊 Download CSV",
                                data=csv_data,
                                file_name=f"{base_filename}_financial_{timestamp}.csv",
                                mime="text/csv",
                                help="Download financial data as CSV file"
                            )
                        
                        with col2:
                            # JSON Export
                            json_data = df.to_json(orient='records', indent=2)
                            
                            st.download_button(
                                label="📋 Download JSON",
                                data=json_data,
                                file_name=f"{base_filename}_financial_{timestamp}.json",
                                mime="application/json",
                                help="Download financial data as JSON"
                            )
                        
                        # Show sample data
                        st.subheader("📋 Sample Financial Data")
                        display_df = df[['page_number', 'word_count', 'char_count', 'table_count', 'has_tables']].head(10)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Show text preview
                        st.subheader("📄 Text Preview")
                        if len(df) > 0:
                            preview_text = df.iloc[0]['cleaned_text'][:1000]
                            st.text_area("First page preview (first 1000 characters):", preview_text, height=200)
                    
                    else:
                        st.warning("⚠️ No financial data could be extracted from the document.")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.error("Please check that you have the required PDF processing libraries installed.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About FinDocPipeline v2.0")
        st.markdown("""
        **Enhanced Features:**
        - 🎙️ **Transcript Recognition**: Extract speaker data from earnings calls
        - 📊 **Financial Processing**: Extract metrics and tables from reports
        - 📈 **Multi-format Support**: Handle presentations and summaries
        - 🔄 **Flexible Export**: CSV and JSON download options
        
        **Supported Document Types:**
        - Financial summaries and reports
        - Earnings presentations
        - Investor meeting transcripts
        - Quarterly earnings calls
        
        **Requirements:**
        - PDF files only
        - Recommended: pdfplumber or PyMuPDF libraries
        """)
        
        st.header("🔧 System Status")
        
        # Check library availability
        try:
            import pdfplumber
            st.success("✅ pdfplumber available")
        except ImportError:
            st.warning("⚠️ pdfplumber not available")
        
        if PYMUPDF_AVAILABLE:
            st.success("✅ PyMuPDF available")
        else:
            st.warning("⚠️ PyMuPDF not available")

if __name__ == "__main__":
    main()