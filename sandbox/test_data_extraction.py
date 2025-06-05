"""
Simple test script to verify text and numeric data extraction is working
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import visual content processing components
from processing.visual_content_processor import VisualContentProcessor
from storage.visual_content_storage import VisualContentStorage

def create_test_table():
    """Create a simple test table"""
    image = np.ones((300, 500, 3), dtype=np.uint8) * 255
    
    # Table data
    headers = ["Quarter", "Revenue", "Profit", "Growth"]
    data = [
        ["Q1 2024", "$75M", "$15M", "12%"],
        ["Q2 2024", "$82M", "$18M", "15%"],
        ["Q3 2024", "$89M", "$21M", "18%"],
        ["Q4 2024", "$95M", "$24M", "20%"]
    ]
    
    # Draw title
    cv2.putText(image, "Quarterly Financial Results", (120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Table dimensions
    cell_width = 100
    cell_height = 40
    start_x = 50
    start_y = 50
    
    # Draw headers
    for i, header in enumerate(headers):
        x = start_x + i * cell_width
        y = start_y
        cv2.rectangle(image, (x, y), (x + cell_width, y + cell_height), (0, 0, 0), 2)
        cv2.putText(image, header, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw data rows
    for row_idx, row in enumerate(data):
        for col_idx, cell in enumerate(row):
            x = start_x + col_idx * cell_width
            y = start_y + (row_idx + 1) * cell_height
            cv2.rectangle(image, (x, y), (x + cell_width, y + cell_height), (0, 0, 0), 1)
            cv2.putText(image, cell, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

def main():
    print("Testing Visual Content Data Extraction")
    print("=" * 50)
    
    # Initialize processor and storage
    processor = VisualContentProcessor()
    storage = VisualContentStorage()
    
    # Create test table
    print("Creating test table...")
    table_image = create_test_table()
    
    # Define test OCR text
    ocr_text = "Quarterly Financial Results\nQuarter\tRevenue\tProfit\tGrowth\nQ1 2024\t$75M\t$15M\t12%\nQ2 2024\t$82M\t$18M\t15%\nQ3 2024\t$89M\t$21M\t18%\nQ4 2024\t$95M\t$24M\t20%"
    
    # Process visual content
    print(" Processing visual content...")
    bounding_box = {'x': 0.1, 'y': 0.1, 'width': 0.8, 'height': 0.8}
    
    visual_content = processor.process_visual_content(
        image=table_image,
        document_id="test_table_001",
        page_number=1,
        bounding_box=bounding_box,
        ocr_text=ocr_text
    )
    
    # Display results
    print("\n Processing Results:")
    print(f"- Visual ID: {visual_content.visual_id}")
    print(f"- Content Type: {visual_content.content_type}")
    print(f"- Table Type: {visual_content.table_type}")
    print(f"- Quality Score: {visual_content.image_quality_score:.3f}")
    print(f"- Confidence: {visual_content.extraction_confidence:.3f}")
    
    # Check extracted data
    print("\n Extracted Data:")
    if visual_content.extracted_data:
        extracted_data = visual_content.extracted_data
        print(f"- Data Quality Score: {extracted_data.data_quality_score:.3f}")
        
        if extracted_data.structured_data:
            structured = extracted_data.structured_data
            print(f"- Headers: {structured.get('headers', [])}")
            print(f"- Number of Rows: {structured.get('num_rows', 0)}")
            print(f"- Number of Columns: {structured.get('num_columns', 0)}")
            
            if 'rows' in structured:
                print("- Sample Data Rows:")
                for i, row in enumerate(structured['rows'][:2]):  # Show first 2 rows
                    print(f"  Row {i+1}: {row}")
        
        if extracted_data.raw_data:
            print(f"- Raw Data Length: {len(extracted_data.raw_data)}")
    else:
        print("- No extracted data found!")
    
    # Check financial metrics
    print("\n Financial Analysis:")
    print(f"- Financial Metrics: {visual_content.financial_metrics}")
    print(f"- Time Periods: {visual_content.time_periods}")
    print(f"- Currencies: {visual_content.currencies}")
    
    # Store and test retrieval
    print("\n Testing Storage...")
    storage.store_visual_content(visual_content)
    
    # Retrieve and verify
    retrieved = storage.get_visual_content(visual_content.visual_id)
    if retrieved:
        print(" Storage and retrieval successful!")
        
        # Test data display
        if retrieved.extracted_data and retrieved.extracted_data.structured_data:
            structured = retrieved.extracted_data.structured_data
            if 'headers' in structured and 'rows' in structured:
                print("\n Retrieved Table Data:")
                headers = structured['headers']
                rows = structured['rows']
                
                # Display as table
                print(f"{'':>10} | " + " | ".join(f"{h:>10}" for h in headers))
                print("-" * (15 + len(headers) * 13))
                for i, row in enumerate(rows):
                    print(f"Row {i+1:>5} | " + " | ".join(f"{cell:>10}" for cell in row))
        else:
            print(" No structured data found in retrieved content!")
    else:
        print(" Storage or retrieval failed!")
    
    print("\n" + "=" * 50)
    print(" Test completed!")

if __name__ == "__main__":
    main()