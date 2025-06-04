import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import re
import subprocess
import sys
import json

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
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\%\$]', ' ', text)
        
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
        
        text_lower = text.lower()
        return any(term in text_lower for term in financial_terms)

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
        
        # Check for table context
        if 'table' in text.lower() or '|' in text:
            confidence += 0.2
        
        # Check for financial statement context
        financial_contexts = ['balance sheet', 'income statement', 'cash flow', 'financial']
        if any(ctx in text.lower() for ctx in financial_contexts):
            confidence += 0.2
        
        # Check for proper formatting
        if '$' in match.group(0) or '£' in match.group(0) or '€' in match.group(0):
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

def main():
    st.set_page_config(
        page_title="BOE ETL - Final Balanced Financial Analysis",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 BOE ETL - Final Balanced Financial Analysis")
    st.markdown("**Three-Tier Data Analysis: Raw Data + NLP-Ready Data + Structured Metrics with Deduplication**")
    
    # Initialize components
    parser = ComprehensiveFinancialParser()
    nlp_processor = NLPDataProcessor()
    metrics_extractor = DeduplicatedMetricsExtractor()
    
    # Check capabilities
    if not parser.pdf_methods:
        st.warning("⚠️ Processing requires pdfplumber or PyMuPDF!")
        if st.button("🔧 Install pdfplumber"):
            if install_pdfplumber():
                st.experimental_rerun()
        return
    else:
        st.success(f"✅ Processing available with: {', '.join(parser.pdf_methods)}")
    
    # File upload
    st.header("📁 Upload Financial Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload financial presentations, earnings reports, or regulatory filings"
    )
    
    if uploaded_file is not None:
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("🔄 Extracting comprehensive data..."):
                    pages_data = parser.extract_comprehensive_data(tmp_path)
                
                with st.spinner("🔄 Creating raw structured data..."):
                    raw_df = nlp_processor.create_raw_csv(pages_data)
                
                with st.spinner("🤖 Cleaning and preparing for NLP..."):
                    nlp_df, full_df = nlp_processor.clean_for_nlp(raw_df)
                
                with st.spinner("🔍 Extracting deduplicated metrics..."):
                    metrics_long_df, debug_df = metrics_extractor.extract_metrics_enhanced(nlp_df)
                    metrics_wide_df = metrics_extractor.create_wide_metrics(metrics_long_df)
                
                st.success(f"✅ Balanced processing finished: {uploaded_file.name} ({len(pages_data)} pages)")
                
                # Display results with balanced layout
                st.header("📊 Three-Tier Data Analysis")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages Processed", len(pages_data))
                with col2:
                    st.metric("NLP-Ready Rows", len(nlp_df))
                with col3:
                    st.metric("Extracted Metrics", len(metrics_long_df))
                with col4:
                    unique_metrics = metrics_long_df['metric_name'].nunique() if len(metrics_long_df) > 0 else 0
                    st.metric("Unique Metric Types", unique_metrics)
                
                # Deduplication Summary
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
                        st.write(f"- Unique metrics after deduplication: {len(metrics_long_df)}")
                        st.write(f"- Metric types found: {unique_metrics}")
                        
                        if len(dedup_stats) > 0:
                            st.write("**Metrics by type:**")
                            for metric, count in dedup_stats.head(5).items():
                                st.write(f"  • {metric}: {count}")
                    else:
                        st.info("No metrics extracted")
                
                # Three-column balanced layout
                st.subheader("📄 Complete Data Preview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**🔤 Raw Data**")
                    st.write("*Original extracted content*")
                    if len(raw_df) > 0:
                        st.dataframe(raw_df.head(10), use_container_width=True, height=400)
                    else:
                        st.info("No raw data available")
                
                with col2:
                    st.write("**🤖 NLP-Ready Data**")
                    st.write("*Cleaned and normalized*")
                    if len(nlp_df) > 0:
                        st.dataframe(nlp_df.head(10), use_container_width=True, height=400)
                    else:
                        st.info("No NLP data available")
                
                with col3:
                    st.write("**📊 Structured Metrics**")
                    st.write("*Extracted financial metrics*")
                    if len(metrics_long_df) > 0:
                        st.dataframe(metrics_long_df.head(10), use_container_width=True, height=400)
                    else:
                        st.info("No metrics extracted")
                
                # Complete Data Preview with all tabs as shown in screenshot
                st.subheader("📄 Complete Data Preview")
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Raw Data", "NLP-Ready", "Metrics (Long)", "Metrics (Wide)", "Debug Info"])
                
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
                    st.write("**Deduplicated metrics in long form (one row per unique metric):**")
                    if len(metrics_long_df) > 0:
                        st.dataframe(metrics_long_df, use_container_width=True)
                    else:
                        st.info("No metrics extracted from this document")
                
                with tab4:
                    st.write("**Extracted metrics in wide form (one row per page):**")
                    if len(metrics_wide_df) > 0:
                        st.dataframe(metrics_wide_df, use_container_width=True)
                    else:
                        st.info("No wide-form metrics available")
                
                with tab5:
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
    
    # Footer
    st.markdown("---")
    st.markdown("**FinDocPipeline** - Standalone Financial Document Analysis Tool")

if __name__ == "__main__":
    main()