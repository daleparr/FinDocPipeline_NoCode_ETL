"""
FinDocPipeline Enhanced - Multi-Document Concurrent Processing
Extends the existing sophisticated single-document system with multi-document capabilities.
Generates detailed row-by-row output like the original system, scaled for multiple documents.
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import existing classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing sophisticated classes
from FinDocPipeline import (
    ComprehensiveFinancialParser,
    NLPDataProcessor, 
    NLPDatasetExporter,
    EnhancedVisualParser,
    DeduplicatedMetricsExtractor
)

import pandas as pd
import tempfile
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

# Import new multi-document components
from file_handlers import BaseFileHandler, PDFHandler, DOCXHandler, ExcelHandler, CSVHandler
from classification.document_classifier import DocumentClassifier
from processing.concurrent_manager import ConcurrentProcessingManager
from schemas.base_schema import BaseDocumentSchema

# Import visual content processing components
from processing.visual_content_extractor import VisualContentExtractor
from storage.visual_content_storage import VisualContentStorage
from ui.visual_analytics_dashboard import VisualAnalyticsDashboard

class MultiDocumentProcessor:
    """
    Multi-document processor that extends existing single-document capabilities
    with concurrent processing and detailed row-by-row output generation.
    """
    
    def __init__(self):
        # Initialize existing processors
        self.pdf_parser = ComprehensiveFinancialParser()
        self.nlp_processor = NLPDataProcessor()
        self.nlp_exporter = NLPDatasetExporter()
        self.visual_parser = EnhancedVisualParser()
        self.metrics_extractor = DeduplicatedMetricsExtractor()
        
        # Initialize new multi-document components
        self.document_classifier = DocumentClassifier()
        self.concurrent_manager = ConcurrentProcessingManager(max_workers=4)
        
        # Initialize visual content processing components
        self.visual_extractor = VisualContentExtractor()
        self.visual_storage = VisualContentStorage()
        self.visual_dashboard = VisualAnalyticsDashboard()
        
        # File handlers
        self.file_handlers = {
            'pdf': PDFHandler(),
            'docx': DOCXHandler(),
            'xlsx': ExcelHandler(),
            'xls': ExcelHandler(),
            'csv': CSVHandler()
        }
        
        # Processing statistics
        self.processing_stats = {
            'total_documents': 0,
            'completed_documents': 0,
            'failed_documents': 0,
            'processing_start_time': None,
            'document_results': {},
            'visual_content_stats': {
                'total_visuals_extracted': 0,
                'charts_found': 0,
                'tables_found': 0,
                'images_found': 0
            }
        }
    
    def process_multiple_documents(self, uploaded_files: List) -> Dict[str, Any]:
        """
        Process multiple documents concurrently generating detailed row-by-row output.
        
        Args:
            uploaded_files: List of Streamlit uploaded files
            
        Returns:
            Dict containing detailed processing results with thousands of rows
        """
        
        if not uploaded_files:
            return {'error': 'No files provided'}
        
        # Initialize processing
        self.processing_stats['total_documents'] = len(uploaded_files)
        self.processing_stats['processing_start_time'] = time.time()
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_container = st.empty()
        results_container = st.empty()
        
        # Process documents in batches for Streamlit compatibility
        batch_size = 5  # Process 5 documents at a time
        all_results = {}
        
        for batch_start in range(0, len(uploaded_files), batch_size):
            batch_end = min(batch_start + batch_size, len(uploaded_files))
            batch = uploaded_files[batch_start:batch_end]
            
            # Update status
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(uploaded_files) + batch_size - 1) // batch_size
            status_container.text(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} documents)..."
            )
            
            # Process batch concurrently
            batch_results = self._process_batch_concurrent(batch)
            all_results.update(batch_results)
            
            # Update progress
            progress = batch_end / len(uploaded_files)
            progress_bar.progress(progress)
            
            # Update results display
            self._update_results_display(results_container, all_results)
            
            # Small delay for UI responsiveness
            time.sleep(0.1)
        
        # Generate consolidated detailed output
        final_results = self._consolidate_detailed_results(all_results)
        
        # Update final status
        processing_time = time.time() - self.processing_stats['processing_start_time']
        status_container.success(f"✅ Completed processing {len(uploaded_files)} documents in {processing_time:.1f} seconds!")
        progress_bar.progress(1.0)
        
        return final_results
    
    def _process_batch_concurrent(self, batch: List) -> Dict[str, Any]:
        """Process a batch of documents concurrently"""
        
        batch_results = {}
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks in the batch
            future_to_file = {}
            for file in batch:
                future = executor.submit(self._process_single_document_detailed, file)
                future_to_file[future] = file
            
            # Collect results as they complete
            for future in as_completed(future_to_file, timeout=120):  # 2 minute timeout
                file = future_to_file[future]
                
                try:
                    result = future.result()
                    batch_results[file.name] = result
                    self.processing_stats['completed_documents'] += 1
                
                except Exception as e:
                    error_result = {
                        'error': str(e),
                        'status': 'failed',
                        'filename': file.name,
                        'file_type': file.type
                    }
                    batch_results[file.name] = error_result
                    self.processing_stats['failed_documents'] += 1
        
        return batch_results
    
    def _process_single_document_detailed(self, uploaded_file) -> Dict[str, Any]:
        """
        Process a single document generating detailed row-by-row output like the original system.
        
        Args:
            uploaded_file: Streamlit uploaded file
            
        Returns:
            Dict containing detailed processing results with thousands of rows
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Classify document type
            classification_result = self.document_classifier.classify_uploaded_file(uploaded_file)
            
            # Step 2: Process based on file type with detailed output
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == 'pdf':
                detailed_result = self._process_pdf_detailed(uploaded_file)
            elif file_ext == 'docx':
                detailed_result = self._process_docx_detailed(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                detailed_result = self._process_excel_detailed(uploaded_file)
            elif file_ext == 'csv':
                detailed_result = self._process_csv_detailed(uploaded_file)
            else:
                detailed_result = {'error': f'Unsupported file type: {file_ext}'}
            
            # Step 3: Combine with classification and metadata
            comprehensive_result = {
                'filename': uploaded_file.name,
                'file_type': uploaded_file.type,
                'file_extension': file_ext,
                'file_size': uploaded_file.size,
                'processing_time': time.time() - start_time,
                'status': 'completed',
                'classification': classification_result,
                'detailed_content': detailed_result,
                'processed_at': datetime.now().isoformat()
            }
            
            return comprehensive_result
        
        except Exception as e:
            return {
                'filename': uploaded_file.name,
                'file_type': uploaded_file.type,
                'error': str(e),
                'status': 'failed',
                'processing_time': time.time() - start_time
            }
    
    def _process_pdf_detailed(self, uploaded_file) -> Dict[str, Any]:
        """Process PDF with detailed row-by-row output using existing sophisticated system"""
        
        try:
            # Save file temporarily for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Use existing ComprehensiveFinancialParser (generates detailed page data)
            pages_data = self.pdf_parser.extract_comprehensive_data(tmp_file_path)
            
            # Use existing NLP processor (generates detailed row data)
            raw_df = self.nlp_processor.create_raw_csv(pages_data)
            nlp_df, _ = self.nlp_processor.clean_for_nlp(raw_df)
            
            # Use existing visual parser (generates detailed visual data)
            visual_data = self.visual_parser.extract_visual_data(tmp_file_path)
            
            # NEW: Extract visual content with embeddings
            uploaded_file.seek(0)  # Reset file pointer
            document_id = f"pdf_{uploaded_file.name}_{int(time.time())}"
            visual_contents = self.visual_extractor.extract_from_pdf(uploaded_file, document_id)
            
            # Store visual contents
            self.visual_extractor.process_and_store_visuals(visual_contents)
            
            # Update visual content statistics
            self._update_visual_stats(visual_contents)
            
            # Use existing metrics extractor (generates detailed metrics)
            metrics_df, debug_df = self.metrics_extractor.extract_metrics_enhanced(nlp_df)
            
            # Create enhanced dataset (generates detailed NLP features)
            enhanced_df = self.visual_parser.create_enhanced_dataset(visual_data, nlp_df)
            
            # Create comprehensive NLP dataset (thousands of detailed rows)
            nlp_dataset = self.nlp_exporter.create_nlp_dataset(enhanced_df, metrics_df)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            # Get visual content extraction summary
            visual_extraction_summary = self.visual_extractor.get_extraction_summary(visual_contents)
            
            return {
                'extraction_method': 'comprehensive_pdf_parser',
                'pages_processed': len(pages_data),
                
                # DETAILED DATA - thousands of rows like original system
                'detailed_pages_data': pages_data,  # Page-by-page detailed extraction
                'detailed_raw_csv': raw_df.to_dict('records'),  # Raw CSV with all content
                'detailed_nlp_data': nlp_df.to_dict('records'),  # Cleaned NLP data
                'detailed_visual_data': visual_data,  # Visual content analysis
                'detailed_metrics': metrics_df.to_dict('records'),  # Financial metrics
                'detailed_debug_info': debug_df.to_dict('records'),  # Debug information
                'detailed_nlp_dataset': nlp_dataset.to_dict('records'),  # Full NLP dataset
                
                # NEW: Visual content with embeddings
                'visual_content_embeddings': [vc.dict() for vc in visual_contents],
                'visual_extraction_summary': visual_extraction_summary,
                
                # Summary statistics
                'summary_stats': {
                    'total_text_length': sum(len(page.get('full_text', '')) for page in pages_data),
                    'total_tables': sum(len(page.get('tables', [])) for page in pages_data),
                    'total_images': sum(len(vd.get('images', [])) for vd in visual_data),
                    'total_charts': sum(len(vd.get('charts_detected', [])) for vd in visual_data),
                    'total_metrics': len(metrics_df),
                    'total_nlp_rows': len(nlp_dataset),
                    'total_raw_rows': len(raw_df),
                    # NEW: Visual content statistics
                    'visual_embeddings_extracted': len(visual_contents),
                    'visual_content_types': visual_extraction_summary.get('content_types', {}),
                    'avg_visual_quality': visual_extraction_summary.get('avg_quality', 0.0)
                }
            }
        
        except Exception as e:
            return {'error': f'PDF detailed processing failed: {str(e)}'}
    
    def _update_visual_stats(self, visual_contents: List) -> None:
        """Update visual content statistics"""
        for visual_content in visual_contents:
            self.processing_stats['visual_content_stats']['total_visuals_extracted'] += 1
            
            content_type = visual_content.content_type
            if content_type == 'chart':
                self.processing_stats['visual_content_stats']['charts_found'] += 1
            elif content_type == 'table':
                self.processing_stats['visual_content_stats']['tables_found'] += 1
            elif content_type == 'image':
                self.processing_stats['visual_content_stats']['images_found'] += 1
    
    def _process_docx_detailed(self, uploaded_file) -> Dict[str, Any]:
        """Process DOCX with detailed output"""
        
        try:
            # Use DOCX handler for extraction
            docx_handler = self.file_handlers['docx']
            extracted_content = docx_handler.extract_content(uploaded_file)
            
            if extracted_content.errors:
                return {'error': f'DOCX processing failed: {"; ".join(extracted_content.errors)}'}
            
            # Create detailed NLP processing for DOCX content
            docx_rows = []
            
            # Split text into paragraphs for detailed analysis
            paragraphs = extracted_content.text.split('\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    docx_rows.append({
                        'document_section': 'paragraph',
                        'section_id': i + 1,
                        'content_type': 'docx_paragraph',
                        'raw_text': paragraph,
                        'word_count': len(paragraph.split()),
                        'char_count': len(paragraph),
                        'extraction_method': 'docx_handler'
                    })
            
            # Add table content as separate rows
            for table_idx, table in enumerate(extracted_content.tables):
                for row_idx, row in enumerate(table.get('data', [])):
                    docx_rows.append({
                        'document_section': 'table',
                        'section_id': f"table_{table_idx}_row_{row_idx}",
                        'content_type': 'docx_table_row',
                        'raw_text': ' | '.join(str(cell) for cell in row),
                        'word_count': len(' | '.join(str(cell) for cell in row).split()),
                        'char_count': len(' | '.join(str(cell) for cell in row)),
                        'extraction_method': 'docx_handler',
                        'table_index': table_idx,
                        'row_index': row_idx
                    })
            
            # Create DataFrame for NLP processing
            docx_df = pd.DataFrame(docx_rows)
            
            # Clean for NLP
            nlp_df, _ = self.nlp_processor.clean_for_nlp(docx_df)
            
            # NEW: Extract visual content with embeddings
            uploaded_file.seek(0)  # Reset file pointer
            document_id = f"docx_{uploaded_file.name}_{int(time.time())}"
            visual_contents = self.visual_extractor.extract_from_docx(uploaded_file, document_id)
            
            # Store visual contents
            self.visual_extractor.process_and_store_visuals(visual_contents)
            
            # Update visual content statistics
            self._update_visual_stats(visual_contents)
            
            # Extract metrics
            metrics_df, debug_df = self.metrics_extractor.extract_metrics_enhanced(nlp_df)
            
            # Create NLP dataset
            nlp_dataset = self.nlp_exporter.create_nlp_dataset(nlp_df, metrics_df)
            
            # Get visual content extraction summary
            visual_extraction_summary = self.visual_extractor.get_extraction_summary(visual_contents)
            
            return {
                'extraction_method': 'docx_detailed_handler',
                
                # DETAILED DATA - row-by-row like original system
                'detailed_raw_data': docx_rows,  # Paragraph and table rows
                'detailed_nlp_data': nlp_df.to_dict('records'),  # NLP processed data
                'detailed_metrics': metrics_df.to_dict('records'),  # Financial metrics
                'detailed_debug_info': debug_df.to_dict('records'),  # Debug information
                'detailed_nlp_dataset': nlp_dataset.to_dict('records'),  # Full NLP dataset
                'detailed_extracted_content': {
                    'text': extracted_content.text,
                    'tables': extracted_content.tables,
                    'structure': extracted_content.structure,
                    'metadata': extracted_content.metadata
                },
                
                # NEW: Visual content with embeddings
                'visual_content_embeddings': [vc.dict() for vc in visual_contents],
                'visual_extraction_summary': visual_extraction_summary,
                
                # Summary statistics
                'summary_stats': {
                    'total_paragraphs': len([r for r in docx_rows if r['document_section'] == 'paragraph']),
                    'total_table_rows': len([r for r in docx_rows if r['document_section'] == 'table']),
                    'total_tables': len(extracted_content.tables),
                    'total_metrics': len(metrics_df),
                    'total_nlp_rows': len(nlp_dataset),
                    'total_detailed_rows': len(docx_rows),
                    # NEW: Visual content statistics
                    'visual_embeddings_extracted': len(visual_contents),
                    'visual_content_types': visual_extraction_summary.get('content_types', {}),
                    'avg_visual_quality': visual_extraction_summary.get('avg_quality', 0.0)
                }
            }
        
        except Exception as e:
            return {'error': f'DOCX detailed processing failed: {str(e)}'}
    
    def _process_excel_detailed(self, uploaded_file) -> Dict[str, Any]:
        """Process Excel with detailed row-by-row output"""
        
        try:
            # Use Excel handler for extraction
            excel_handler = self.file_handlers['xlsx']
            extracted_content = excel_handler.extract_content(uploaded_file)
            
            if extracted_content.errors:
                return {'error': f'Excel processing failed: {"; ".join(extracted_content.errors)}'}
            
            # Create detailed row-by-row data
            excel_rows = []
            
            # Process each table (sheet) in detail
            for table_idx, table in enumerate(extracted_content.tables):
                sheet_name = table.get('sheet_name', f'Sheet_{table_idx}')
                
                # Process each row in the sheet
                for row_idx, row in enumerate(table.get('data', [])):
                    excel_rows.append({
                        'document_section': 'excel_sheet',
                        'sheet_name': sheet_name,
                        'section_id': f"{sheet_name}_row_{row_idx}",
                        'content_type': 'excel_row',
                        'raw_text': ' | '.join(str(cell) for cell in row if cell is not None),
                        'word_count': len(' | '.join(str(cell) for cell in row if cell is not None).split()),
                        'char_count': len(' | '.join(str(cell) for cell in row if cell is not None)),
                        'extraction_method': 'excel_handler',
                        'sheet_index': table_idx,
                        'row_index': row_idx,
                        'column_count': len(row)
                    })
            
            # Create DataFrame for NLP processing
            excel_df = pd.DataFrame(excel_rows)
            
            # Clean for NLP
            nlp_df, _ = self.nlp_processor.clean_for_nlp(excel_df)
            
            # NEW: Extract visual content with embeddings
            uploaded_file.seek(0)  # Reset file pointer
            document_id = f"excel_{uploaded_file.name}_{int(time.time())}"
            visual_contents = self.visual_extractor.extract_from_excel(uploaded_file, document_id)
            
            # Store visual contents
            self.visual_extractor.process_and_store_visuals(visual_contents)
            
            # Update visual content statistics
            self._update_visual_stats(visual_contents)
            
            # Extract metrics
            metrics_df, debug_df = self.metrics_extractor.extract_metrics_enhanced(nlp_df)
            
            # Create NLP dataset
            nlp_dataset = self.nlp_exporter.create_nlp_dataset(nlp_df, metrics_df)
            
            # Get visual content extraction summary
            visual_extraction_summary = self.visual_extractor.get_extraction_summary(visual_contents)
            
            return {
                'extraction_method': 'excel_detailed_handler',
                
                # DETAILED DATA - row-by-row processing
                'detailed_raw_data': excel_rows,  # Every Excel row as separate entry
                'detailed_nlp_data': nlp_df.to_dict('records'),  # NLP processed data
                'detailed_metrics': metrics_df.to_dict('records'),  # Financial metrics
                'detailed_debug_info': debug_df.to_dict('records'),  # Debug information
                'detailed_nlp_dataset': nlp_dataset.to_dict('records'),  # Full NLP dataset
                'detailed_extracted_content': {
                    'text': extracted_content.text,
                    'tables': extracted_content.tables,
                    'structure': extracted_content.structure,
                    'metadata': extracted_content.metadata
                },
                
                # NEW: Visual content with embeddings
                'visual_content_embeddings': [vc.dict() for vc in visual_contents],
                'visual_extraction_summary': visual_extraction_summary,
                
                # Summary statistics
                'summary_stats': {
                    'total_sheets': len(extracted_content.tables),
                    'total_rows': len(excel_rows),
                    'total_metrics': len(metrics_df),
                    'total_nlp_rows': len(nlp_dataset),
                    'sheets_processed': [t.get('sheet_name', f'Sheet_{i}') for i, t in enumerate(extracted_content.tables)],
                    # NEW: Visual content statistics
                    'visual_embeddings_extracted': len(visual_contents),
                    'visual_content_types': visual_extraction_summary.get('content_types', {}),
                    'avg_visual_quality': visual_extraction_summary.get('avg_quality', 0.0)
                }
            }
        
        except Exception as e:
            return {'error': f'Excel detailed processing failed: {str(e)}'}
    
    def _process_csv_detailed(self, uploaded_file) -> Dict[str, Any]:
        """Process CSV with detailed row-by-row output"""
        
        try:
            # Use CSV handler for extraction
            csv_handler = self.file_handlers['csv']
            extracted_content = csv_handler.extract_content(uploaded_file)
            
            if extracted_content.errors:
                return {'error': f'CSV processing failed: {"; ".join(extracted_content.errors)}'}
            
            # Create detailed row-by-row data
            csv_rows = []
            
            # Process each row in the CSV
            if extracted_content.tables:
                table = extracted_content.tables[0]  # CSV has one table
                for row_idx, row in enumerate(table.get('data', [])):
                    csv_rows.append({
                        'document_section': 'csv_data',
                        'section_id': f"row_{row_idx}",
                        'content_type': 'csv_row',
                        'raw_text': ' | '.join(str(cell) for cell in row if cell is not None),
                        'word_count': len(' | '.join(str(cell) for cell in row if cell is not None).split()),
                        'char_count': len(' | '.join(str(cell) for cell in row if cell is not None)),
                        'extraction_method': 'csv_handler',
                        'row_index': row_idx,
                        'column_count': len(row)
                    })
            
            # Create DataFrame for NLP processing
            csv_df = pd.DataFrame(csv_rows)
            
            # Clean for NLP
            nlp_df, _ = self.nlp_processor.clean_for_nlp(csv_df)
            
            # Extract metrics
            metrics_df, debug_df = self.metrics_extractor.extract_metrics_enhanced(nlp_df)
            
            # Create NLP dataset
            nlp_dataset = self.nlp_exporter.create_nlp_dataset(nlp_df, metrics_df)
            
            return {
                'extraction_method': 'csv_detailed_handler',
                
                # DETAILED DATA - every CSV row processed
                'detailed_raw_data': csv_rows,  # Every CSV row as separate entry
                'detailed_nlp_data': nlp_df.to_dict('records'),  # NLP processed data
                'detailed_metrics': metrics_df.to_dict('records'),  # Financial metrics
                'detailed_debug_info': debug_df.to_dict('records'),  # Debug information
                'detailed_nlp_dataset': nlp_dataset.to_dict('records'),  # Full NLP dataset
                'detailed_extracted_content': {
                    'text': extracted_content.text,
                    'tables': extracted_content.tables,
                    'structure': extracted_content.structure,
                    'metadata': extracted_content.metadata
                },
                
                # Summary statistics
                'summary_stats': {
                    'total_rows': len(csv_rows),
                    'total_columns': extracted_content.tables[0].get('cols', 0) if extracted_content.tables else 0,
                    'total_metrics': len(metrics_df),
                    'total_nlp_rows': len(nlp_dataset)
                }
            }
        
        except Exception as e:
            return {'error': f'CSV detailed processing failed: {str(e)}'}
    
    def _consolidate_detailed_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate all detailed results maintaining row-level granularity"""
        
        # Calculate summary statistics
        total_docs = len(all_results)
        successful_docs = sum(1 for result in all_results.values() if result.get('status') == 'completed')
        failed_docs = total_docs - successful_docs
        
        # Aggregate detailed data across all documents
        consolidated_detailed_data = {
            'all_detailed_pages': [],
            'all_detailed_raw_data': [],
            'all_detailed_nlp_data': [],
            'all_detailed_metrics': [],
            'all_detailed_nlp_dataset': [],
            'all_detailed_visual_data': []
        }
        
        # Aggregate counts
        total_detailed_rows = 0
        total_nlp_rows = 0
        total_metrics = 0
        
        for filename, result in all_results.items():
            if result.get('status') == 'completed':
                detailed_content = result.get('detailed_content', {})
                
                # Add document identifier to each row
                doc_id = filename
                
                # Aggregate detailed pages data (for PDFs)
                if 'detailed_pages_data' in detailed_content:
                    for page in detailed_content['detailed_pages_data']:
                        page['source_document'] = doc_id
                        consolidated_detailed_data['all_detailed_pages'].append(page)
                
                # Aggregate detailed raw data
                if 'detailed_raw_data' in detailed_content:
                    for row in detailed_content['detailed_raw_data']:
                        row['source_document'] = doc_id
                        consolidated_detailed_data['all_detailed_raw_data'].append(row)
                        total_detailed_rows += 1
                
                # Aggregate detailed NLP data
                if 'detailed_nlp_data' in detailed_content:
                    for row in detailed_content['detailed_nlp_data']:
                        row['source_document'] = doc_id
                        consolidated_detailed_data['all_detailed_nlp_data'].append(row)
                
                # Aggregate detailed metrics
                if 'detailed_metrics' in detailed_content:
                    for row in detailed_content['detailed_metrics']:
                        row['source_document'] = doc_id
                        consolidated_detailed_data['all_detailed_metrics'].append(row)
                        total_metrics += 1
                
                # Aggregate detailed NLP dataset
                if 'detailed_nlp_dataset' in detailed_content:
                    for row in detailed_content['detailed_nlp_dataset']:
                        row['source_document'] = doc_id
                        consolidated_detailed_data['all_detailed_nlp_dataset'].append(row)
                        total_nlp_rows += 1
                
                # Aggregate visual data
                if 'detailed_visual_data' in detailed_content:
                    for visual in detailed_content['detailed_visual_data']:
                        visual['source_document'] = doc_id
                        consolidated_detailed_data['all_detailed_visual_data'].append(visual)
        
        # Calculate processing time
        total_processing_time = time.time() - self.processing_stats['processing_start_time']
        
        return {
            'processing_summary': {
                'total_documents': total_docs,
                'successful_documents': successful_docs,
                'failed_documents': failed_docs,
                'success_rate': successful_docs / total_docs if total_docs > 0 else 0,
                'total_processing_time': total_processing_time,
                'average_processing_time': total_processing_time / total_docs if total_docs > 0 else 0,
                'total_detailed_rows_generated': total_detailed_rows,
                'total_nlp_rows_generated': total_nlp_rows,
                'total_metrics_extracted': total_metrics
            },
            
            # CONSOLIDATED DETAILED DATA - thousands of rows across all documents
            'consolidated_detailed_data': consolidated_detailed_data,
            
            # Individual document results (for reference)
            'individual_document_results': all_results,
            
            'processed_at': datetime.now().isoformat()
        }
    
    def _update_results_display(self, container, results: Dict[str, Any]):
        """Update real-time results display"""
        
        if not results:
            return
        
        # Count status
        completed = sum(1 for r in results.values() if r.get('status') == 'completed')
        failed = sum(1 for r in results.values() if r.get('status') == 'failed')
        total = len(results)
        
        with container.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Completed", completed, f"{completed/total:.1%}" if total > 0 else "0%")
            
            with col2:
                st.metric("Failed", failed, f"{failed/total:.1%}" if total > 0 else "0%")
            
            with col3:
                st.metric("Total", total)
            
            # Show recent results
            if results:
                st.subheader("Recent Results")
                recent_results = list(results.items())[-3:]  # Show last 3
                
                for filename, result in recent_results:
                    status = result.get('status', 'unknown')
                    if status == 'completed':
                        # Show detailed row counts
                        detailed_content = result.get('detailed_content', {})
                        summary_stats = detailed_content.get('summary_stats', {})
                        
                        total_rows = (summary_stats.get('total_detailed_rows', 0) + 
                                    summary_stats.get('total_nlp_rows', 0) + 
                                    summary_stats.get('total_raw_rows', 0))
                        
                        st.success(f"✅ {filename} - Generated {total_rows:,} detailed rows")
                    elif status == 'failed':
                        st.error(f"❌ {filename}: {result.get('error', 'Unknown error')}")

def main():
    st.set_page_config(
        page_title="FinDocPipeline Enhanced - Multi-Document Processing",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("📊 FinDocPipeline Enhanced")
    st.markdown("**Multi-Document Concurrent Processing with Detailed Row-by-Row Output**")
    st.markdown("*Generates thousands of detailed rows like the original system, scaled for multiple documents*")
    
    # Initialize processor
    if 'multi_processor' not in st.session_state:
        st.session_state.multi_processor = MultiDocumentProcessor()
    
    processor = st.session_state.multi_processor
    
    # File upload interface with ALL supported formats
    st.header("📁 Upload Multiple Documents")
    st.markdown("*Support for PDF, DOCX, Excel (XLSX/XLS), and CSV files - Up to 20 documents*")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'xlsx', 'xls', 'csv'],  # Added xlsx, xls, csv
        accept_multiple_files=True,
        help="Upload financial documents for concurrent detailed processing"
    )
    
    if uploaded_files:
        # Validate file count
        if len(uploaded_files) > 20:
            st.error("⚠️ Maximum 20 files allowed. Please remove some files.")
            uploaded_files = uploaded_files[:20]
        
        # Display file queue with file types
        st.subheader(f"📋 Document Queue ({len(uploaded_files)} files)")
        
        # Show file details
        for i, file in enumerate(uploaded_files):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{file.name}**")
            with col2:
                st.write(f"{file.size / 1024:.1f} KB")
            with col3:
                file_ext = file.name.split('.')[-1].upper()
                st.write(f"**{file_ext}**")
            with col4:
                # Show file type icon
                type_icons = {
                    'PDF': '📄', 'DOCX': '📝', 'XLSX': '📊', 
                    'XLS': '📊', 'CSV': '📋'
                }
                st.write(type_icons.get(file_ext, '📄'))
        
        # Processing controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Process All Documents (Detailed Output)"):
                with st.spinner("Processing documents with detailed row-by-row analysis..."):
                    results = processor.process_multiple_documents(uploaded_files)
                    st.session_state.processing_results = results
        
        with col2:
            if st.button("📊 Show File Details"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"**{file.name}** - {file.size / 1024:.1f} KB - {file.type}")
        
        with col3:
            if st.button("🗑️ Clear Queue"):
                st.rerun()
    
    # Show processing results with detailed data
    if 'processing_results' in st.session_state:
            results = st.session_state.processing_results
            
            st.header("📈 Detailed Processing Results")
            
            # Processing summary
            summary = results.get('processing_summary', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", summary.get('total_documents', 0))
            with col2:
                st.metric("Successful", summary.get('successful_documents', 0))
            with col3:
                st.metric("Failed", summary.get('failed_documents', 0))
            with col4:
                st.metric("Success Rate", f"{summary.get('success_rate', 0):.1%}")
            
            # Detailed row counts - KEY FEATURE
            st.subheader("📊 Detailed Data Generated (Like Original System)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Detailed Rows", f"{summary.get('total_detailed_rows_generated', 0):,}")
            with col2:
                st.metric("NLP Dataset Rows", f"{summary.get('total_nlp_rows_generated', 0):,}")
            with col3:
                st.metric("Financial Metrics", f"{summary.get('total_metrics_extracted', 0):,}")
            
            # Export consolidated detailed data
            st.subheader("💾 Export Detailed Data")
            consolidated_data = results.get('consolidated_detailed_data', {})
            
            if consolidated_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    raw_data = consolidated_data.get('all_detailed_raw_data', [])
                    if raw_data and st.button("📄 Export All Raw Data CSV"):
                        raw_df = pd.DataFrame(raw_data)
                        csv_data = raw_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {len(raw_data):,} Raw Data Rows",
                            data=csv_data,
                            file_name=f"multi_doc_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    nlp_data = consolidated_data.get('all_detailed_nlp_data', [])
                    if nlp_data and st.button("📊 Export All NLP Data CSV"):
                        nlp_df = pd.DataFrame(nlp_data)
                        csv_data = nlp_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {len(nlp_data):,} NLP Rows",
                            data=csv_data,
                            file_name=f"multi_doc_nlp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    metrics_data = consolidated_data.get('all_detailed_metrics', [])
                    if metrics_data and st.button("💰 Export All Metrics CSV"):
                        metrics_df = pd.DataFrame(metrics_data)
                        csv_data = metrics_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {len(metrics_data):,} Metrics",
                            data=csv_data,
                            file_name=f"multi_doc_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            # Individual document results with detailed data
            st.subheader("📄 Individual Document Results")
            
            document_results = results.get('individual_document_results', {})
            if document_results:
                for filename, doc_result in document_results.items():
                    if doc_result.get('status') == 'completed':
                        detailed_content = doc_result.get('detailed_content', {})
                        summary_stats = detailed_content.get('summary_stats', {})
                        
                        # Calculate total rows for this document
                        total_rows = (
                            len(detailed_content.get('detailed_raw_data', [])) +
                            len(detailed_content.get('detailed_nlp_data', [])) +
                            len(detailed_content.get('detailed_nlp_dataset', []))
                        )
                        
                        with st.expander(f"📄 {filename} - {doc_result.get('classification', {}).get('document_type', 'Unknown')} ({total_rows:,} rows)"):
                            
                            # Document processing stats
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Processing Time", f"{doc_result.get('processing_time', 0):.2f}s")
                            with col2:
                                st.metric("Raw Data Rows", f"{len(detailed_content.get('detailed_raw_data', [])):,}")
                            with col3:
                                st.metric("NLP Rows", f"{len(detailed_content.get('detailed_nlp_data', [])):,}")
                            with col4:
                                st.metric("Metrics Found", f"{len(detailed_content.get('detailed_metrics', [])):,}")
                            
                            # Show sample of detailed data
                            tabs = st.tabs(["📊 Raw Data Sample", "🧠 NLP Data Sample", "💰 Metrics Sample", "📈 Visual Data"])
                            
                            with tabs[0]:
                                raw_data = detailed_content.get('detailed_raw_data', [])
                                if raw_data:
                                    st.write(f"**Showing first 10 of {len(raw_data):,} detailed rows:**")
                                    sample_df = pd.DataFrame(raw_data[:10])
                                    st.dataframe(sample_df, use_container_width=True)
                                    
                                    # Export individual document raw data
                                    if st.button(f"📄 Export {filename} Raw Data", key=f"raw_{filename}"):
                                        full_df = pd.DataFrame(raw_data)
                                        csv_data = full_df.to_csv(index=False)
                                        st.download_button(
                                            label=f"Download {len(raw_data):,} rows",
                                            data=csv_data,
                                            file_name=f"{filename}_raw_data.csv",
                                            mime="text/csv",
                                            key=f"raw_download_{filename}"
                                        )
                                else:
                                    st.info("No raw data available for this document")
                            
                            with tabs[1]:
                                nlp_data = detailed_content.get('detailed_nlp_data', [])
                                if nlp_data:
                                    st.write(f"**Showing first 10 of {len(nlp_data):,} NLP rows:**")
                                    sample_df = pd.DataFrame(nlp_data[:10])
                                    st.dataframe(sample_df, use_container_width=True)
                                    
                                    # Export individual document NLP data
                                    if st.button(f"🧠 Export {filename} NLP Data", key=f"nlp_{filename}"):
                                        full_df = pd.DataFrame(nlp_data)
                                        csv_data = full_df.to_csv(index=False)
                                        st.download_button(
                                            label=f"Download {len(nlp_data):,} rows",
                                            data=csv_data,
                                            file_name=f"{filename}_nlp_data.csv",
                                            mime="text/csv",
                                            key=f"nlp_download_{filename}"
                                        )
                                else:
                                    st.info("No NLP data available for this document")
                            
                            with tabs[2]:
                                metrics_data = detailed_content.get('detailed_metrics', [])
                                if metrics_data:
                                    st.write(f"**Showing all {len(metrics_data):,} metrics:**")
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df, use_container_width=True)
                                    
                                    # Export individual document metrics
                                    if st.button(f"💰 Export {filename} Metrics", key=f"metrics_{filename}"):
                                        csv_data = metrics_df.to_csv(index=False)
                                        st.download_button(
                                            label=f"Download {len(metrics_data):,} metrics",
                                            data=csv_data,
                                            file_name=f"{filename}_metrics.csv",
                                            mime="text/csv",
                                            key=f"metrics_download_{filename}"
                                        )
                                else:
                                    st.info("No metrics available for this document")
                            
                            with tabs[3]:
                                visual_data = detailed_content.get('detailed_visual_data', {})
                                if visual_data:
                                    st.write("**Visual Analysis Results:**")
                                    st.json(visual_data)
                                else:
                                    st.info("No visual data available for this document")
                    
                    else:
                        # Show failed documents
                        with st.expander(f"❌ {filename} - FAILED"):
                            st.error(f"Error: {doc_result.get('error', 'Unknown error')}")
            
            # Processing logs
            if st.checkbox("Show Processing Logs"):
                logs = results.get('processing_logs', [])
                if logs:
                    st.subheader("📋 Processing Logs")
                    for log in logs[-20:]:  # Show last 20 logs
                        st.text(log)
                else:
                    st.info("No processing logs available")
    
    # Visual Content Analytics Dashboard
    st.header("🖼️ Visual Content Analytics")
    st.markdown("*Comprehensive analysis of extracted charts, graphs, tables, and images with semantic embeddings*")
    
    # Check if there are any visual contents stored
    visual_stats = processor.visual_storage.get_visual_statistics()
    
    if visual_stats['total_visuals'] > 0:
        # Show visual content statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Visual Elements", visual_stats['total_visuals'])
        
        with col2:
            charts_count = visual_stats['content_types'].get('chart', 0)
            st.metric("Charts Found", charts_count)
        
        with col3:
            tables_count = visual_stats['content_types'].get('table', 0)
            st.metric("Tables Found", tables_count)
        
        with col4:
            avg_quality = visual_stats['quality_stats'].get('avg_quality', 0)
            st.metric("Avg Quality Score", f"{avg_quality:.2f}")
        
        # Render the visual analytics dashboard
        processor.visual_dashboard.render_dashboard()
        
    else:
        st.info("📊 No visual content has been processed yet. Upload and process documents to see visual analytics.")
        
        # Show what visual content analysis provides
        st.markdown("""
        ### 🎯 Visual Content Analysis Features:
        
        **📊 Chart & Graph Analysis:**
        - Automatic detection of line charts, bar charts, pie charts, scatter plots
        - Data extraction from chart images using OCR
        - Financial metrics identification (revenue, profit, growth rates)
        - Time period recognition (Q1, Q2, annual, monthly)
        
        **📋 Table Processing:**
        - Table structure recognition and data extraction
        - Header identification and data organization
        - Financial statement analysis
        - Comparison table processing
        
        **🖼️ Image Embeddings:**
        - Semantic embeddings for visual similarity search
        - Perceptual hashing for duplicate detection
        - Quality assessment and confidence scoring
        - Content type classification
        
        **🔍 Advanced Analytics:**
        - Visual similarity clustering
        - Cross-document visual comparison
        - Quality metrics and processing statistics
        - Export capabilities for further analysis
        """)


if __name__ == "__main__":
    main()