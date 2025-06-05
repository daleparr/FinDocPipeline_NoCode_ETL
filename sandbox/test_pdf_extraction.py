"""
Test script to verify PDF visual content extraction is working with real documents
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import tempfile

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import visual content processing components
from processing.visual_content_extractor import VisualContentExtractor
from processing.visual_content_processor import VisualContentProcessor
from storage.visual_content_storage import VisualContentStorage

def create_test_pdf():
    """Create a simple test PDF with table content"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        
        # Create temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        temp_file.close()
        
        # Create PDF document
        doc = SimpleDocTemplate(temp_path, pagesize=letter)
        elements = []
        
        # Add title
        styles = getSampleStyleSheet()
        title = Paragraph("Financial Performance Report Q1-Q4 2024", styles['Title'])
        elements.append(title)
        
        # Add table data
        data = [
            ['Quarter', 'Revenue', 'Profit', 'Growth'],
            ['Q1 2024', '$75M', '$15M', '12%'],
            ['Q2 2024', '$82M', '$18M', '15%'],
            ['Q3 2024', '$89M', '$21M', '18%'],
            ['Q4 2024', '$95M', '$24M', '20%']
        ]
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        
        return temp_path
        
    except ImportError:
        print("ReportLab not available, creating simple text-based test")
        return None

def main():
    print("Testing PDF Visual Content Extraction")
    print("=" * 50)
    
    # Initialize components
    extractor = VisualContentExtractor()
    processor = VisualContentProcessor()
    storage = VisualContentStorage()
    
    # Create test PDF
    print("Creating test PDF...")
    pdf_path = create_test_pdf()
    
    if pdf_path:
        print(f"Test PDF created: {pdf_path}")
        
        # Test extraction
        print("Extracting visual content from PDF...")
        
        try:
            # Create a file-like object for testing
            with open(pdf_path, 'rb') as f:
                class MockUploadedFile:
                    def __init__(self, file_obj):
                        self.file_obj = file_obj
                        self.name = os.path.basename(pdf_path)
                        
                    def getvalue(self):
                        return self.file_obj.read()
                
                mock_file = MockUploadedFile(f)
                
                # Extract visual content
                visual_contents = extractor.extract_from_pdf(mock_file, "test_pdf_001")
                
                print(f"Extracted {len(visual_contents)} visual elements")
                
                # Display results
                for i, visual in enumerate(visual_contents):
                    print(f"\nVisual {i+1}:")
                    print(f"- Visual ID: {visual.visual_id}")
                    print(f"- Content Type: {visual.content_type}")
                    print(f"- Page: {visual.page_number}")
                    print(f"- Quality: {visual.image_quality_score:.3f}")
                    print(f"- Confidence: {visual.extraction_confidence:.3f}")
                    
                    if visual.extracted_data:
                        print(f"- Data Quality: {visual.extracted_data.data_quality_score:.3f}")
                        if visual.extracted_data.structured_data:
                            structured = visual.extracted_data.structured_data
                            print(f"- Structured Data Keys: {list(structured.keys())}")
                            if 'headers' in structured and structured['headers']:
                                print(f"- Headers: {structured['headers']}")
                            if 'rows' in structured and structured['rows']:
                                print(f"- Rows: {len(structured['rows'])}")
                                if structured['rows']:
                                    print(f"- Sample Row: {structured['rows'][0]}")
                        if visual.extracted_data.raw_data:
                            print(f"- Raw Data: {len(visual.extracted_data.raw_data)} items")
                    else:
                        print("- No extracted data")
                    
                    print(f"- Financial Metrics: {visual.financial_metrics}")
                    print(f"- Time Periods: {visual.time_periods}")
                    print(f"- Currencies: {visual.currencies}")
                
                # Clean up
                os.unlink(pdf_path)
                
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        print("Could not create test PDF. Testing with fallback method...")
        
        # Test with a simple image-based approach
        print("Testing with synthetic page image...")
        
        # Create a synthetic page image with table content
        page_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Draw title
        cv2.putText(page_image, "Financial Report Q1-Q4 2024", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Draw table structure
        headers = ["Quarter", "Revenue", "Profit", "Growth"]
        data = [
            ["Q1 2024", "$75M", "$15M", "12%"],
            ["Q2 2024", "$82M", "$18M", "15%"],
            ["Q3 2024", "$89M", "$21M", "18%"],
            ["Q4 2024", "$95M", "$24M", "20%"]
        ]
        
        # Table dimensions
        cell_width = 120
        cell_height = 40
        start_x = 50
        start_y = 100
        
        # Draw headers
        for i, header in enumerate(headers):
            x = start_x + i * cell_width
            y = start_y
            cv2.rectangle(page_image, (x, y), (x + cell_width, y + cell_height), (0, 0, 0), 2)
            cv2.putText(page_image, header, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw data rows
        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                x = start_x + col_idx * cell_width
                y = start_y + (row_idx + 1) * cell_height
                cv2.rectangle(page_image, (x, y), (x + cell_width, y + cell_height), (0, 0, 0), 1)
                cv2.putText(page_image, cell, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Test the region splitting
        page_text = "Financial Report Q1-Q4 2024\nQuarter\tRevenue\tProfit\tGrowth\nQ1 2024\t$75M\t$15M\t12%\nQ2 2024\t$82M\t$18M\t15%\nQ3 2024\t$89M\t$21M\t18%\nQ4 2024\t$95M\t$24M\t20%"
        
        regions = extractor._split_page_into_regions(page_image, page_text)
        
        print(f"Split page into {len(regions)} regions")
        
        # Process each region
        for i, region in enumerate(regions):
            print(f"\nRegion {i+1} ({region['region_id']}):")
            print(f"- Image size: {region['image'].shape}")
            print(f"- Bounding box: {region['bbox']}")
            print(f"- Text length: {len(region['text'])}")
            
            # Process with visual content processor
            visual_content = processor.process_visual_content(
                image=region['image'],
                document_id="test_synthetic_001",
                page_number=1,
                bounding_box=region['bbox'],
                ocr_text=region['text'],
                document_name="test_synthetic_document.pdf",
                page_context=f"Page 1 Region {i+1}"
            )
            
            print(f"- Content Type: {visual_content.content_type}")
            print(f"- Quality: {visual_content.image_quality_score:.3f}")
            
            if visual_content.extracted_data and visual_content.extracted_data.structured_data:
                structured = visual_content.extracted_data.structured_data
                if 'headers' in structured and structured['headers']:
                    print(f"- Headers: {structured['headers']}")
                if 'rows' in structured and structured['rows']:
                    print(f"- Data rows: {len(structured['rows'])}")
    
    print("\n" + "=" * 50)
    print("PDF extraction test completed!")

if __name__ == "__main__":
    main()