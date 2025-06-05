"""
Test script for Visual Content Processing in FinDocPipeline
Demonstrates image/chart/graph/table embeddings and analysis capabilities.
"""

import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image
import io
import base64
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import visual content processing components
from processing.visual_content_processor import VisualContentProcessor
from processing.visual_content_extractor import VisualContentExtractor
from storage.visual_content_storage import VisualContentStorage
from ui.visual_analytics_dashboard import VisualAnalyticsDashboard

def create_sample_chart():
    """Create a sample chart image for testing"""
    
    # Create a simple line chart
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(image, (80, 350), (550, 350), (0, 0, 0), 2)  # X-axis
    cv2.line(image, (80, 50), (80, 350), (0, 0, 0), 2)    # Y-axis
    
    # Draw title
    cv2.putText(image, "Revenue Growth 2020-2024", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Draw data points and lines
    points = [(120, 300), (200, 250), (280, 200), (360, 150), (440, 100)]
    values = ["2020: $100M", "2021: $150M", "2022: $200M", "2023: $250M", "2024: $300M"]
    
    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], (255, 0, 0), 3)
        cv2.circle(image, points[i], 5, (0, 0, 255), -1)
    
    cv2.circle(image, points[-1], 5, (0, 0, 255), -1)
    
    # Add labels
    for i, (point, value) in enumerate(zip(points, values)):
        cv2.putText(image, f"{2020+i}", (point[0]-10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, value, (point[0]-30, point[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
    
    # Add axis labels
    cv2.putText(image, "Year", (300, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, "Revenue", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return image

def create_sample_table():
    """Create a sample table image for testing"""
    
    # Create table image
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
        
        # Draw cell border
        cv2.rectangle(image, (x, y), (x + cell_width, y + cell_height), (0, 0, 0), 2)
        
        # Add text
        cv2.putText(image, header, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw data rows
    for row_idx, row in enumerate(data):
        for col_idx, cell in enumerate(row):
            x = start_x + col_idx * cell_width
            y = start_y + (row_idx + 1) * cell_height
            
            # Draw cell border
            cv2.rectangle(image, (x, y), (x + cell_width, y + cell_height), (0, 0, 0), 1)
            
            # Add text
            cv2.putText(image, cell, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

def create_sample_pie_chart():
    """Create a sample pie chart image for testing"""
    
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw title
    cv2.putText(image, "Market Share Distribution", (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Pie chart center and radius
    center = (200, 200)
    radius = 100
    
    # Data for pie chart
    segments = [
        {"label": "Product A: 40%", "angle": 144, "color": (255, 100, 100)},
        {"label": "Product B: 30%", "angle": 108, "color": (100, 255, 100)},
        {"label": "Product C: 20%", "angle": 72, "color": (100, 100, 255)},
        {"label": "Others: 10%", "angle": 36, "color": (255, 255, 100)}
    ]
    
    # Draw pie segments
    start_angle = 0
    for segment in segments:
        end_angle = start_angle + segment["angle"]
        
        # Create points for the segment
        points = [center]
        for angle in range(start_angle, end_angle + 1, 5):
            x = int(center[0] + radius * np.cos(np.radians(angle)))
            y = int(center[1] + radius * np.sin(np.radians(angle)))
            points.append((x, y))
        
        # Draw filled polygon
        points_array = np.array(points, np.int32)
        cv2.fillPoly(image, [points_array], segment["color"])
        
        # Draw border
        cv2.polylines(image, [points_array], True, (0, 0, 0), 2)
        
        start_angle = end_angle
    
    # Add legend
    legend_y = 320
    for i, segment in enumerate(segments):
        y = legend_y + i * 20
        cv2.rectangle(image, (50, y), (70, y + 15), segment["color"], -1)
        cv2.rectangle(image, (50, y), (70, y + 15), (0, 0, 0), 1)
        cv2.putText(image, segment["label"], (80, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

def main():
    st.set_page_config(
        page_title="Visual Content Processing Test",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Visual Content Processing Test")
    st.markdown("**Testing image/chart/graph/table embeddings and analysis capabilities**")
    
    # Initialize components
    if 'visual_processor' not in st.session_state:
        st.session_state.visual_processor = VisualContentProcessor()
        st.session_state.visual_storage = VisualContentStorage()
        st.session_state.visual_dashboard = VisualAnalyticsDashboard()
        st.session_state.visual_extractor = VisualContentExtractor()
    
    processor = st.session_state.visual_processor
    storage = st.session_state.visual_storage
    dashboard = st.session_state.visual_dashboard
    extractor = st.session_state.visual_extractor
    
    # Standard Document Upload Interface (matching existing pipeline)
    st.header("📁 Upload Documents for Visual Content Analysis")
    st.markdown("*Upload PDF, DOCX, or Excel files to extract and analyze visual content*")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload financial documents for visual content extraction and embedding generation"
    )
    
    if uploaded_files:
        st.subheader(f"📋 Document Queue ({len(uploaded_files)} files)")
        
        # Show file details (matching existing pipeline style)
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
                type_icons = {'PDF': '📄', 'DOCX': '📝', 'XLSX': '📊', 'XLS': '📊'}
                st.write(type_icons.get(file_ext, '📄'))
        
        # Process documents
        if st.button("🚀 Extract Visual Content from Documents"):
            with st.spinner("Extracting visual content and generating embeddings..."):
                
                total_visuals = 0
                processing_results = {}
                
                for file in uploaded_files:
                    try:
                        file_ext = file.name.split('.')[-1].lower()
                        document_id = f"{file_ext}_{file.name}_{int(time.time())}"
                        
                        # Extract visual content based on file type
                        if file_ext == 'pdf':
                            visual_contents = extractor.extract_from_pdf(file, document_id)
                        elif file_ext == 'docx':
                            visual_contents = extractor.extract_from_docx(file, document_id)
                        elif file_ext in ['xlsx', 'xls']:
                            visual_contents = extractor.extract_from_excel(file, document_id)
                        else:
                            continue
                        
                        # Store visual contents
                        extractor.process_and_store_visuals(visual_contents)
                        
                        # Get extraction summary
                        summary = extractor.get_extraction_summary(visual_contents)
                        processing_results[file.name] = {
                            'visual_contents': visual_contents,
                            'summary': summary
                        }
                        
                        total_visuals += len(visual_contents)
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                        processing_results[file.name] = {'error': str(e)}
                
                st.success(f"✅ Extracted {total_visuals} visual elements from {len(uploaded_files)} documents!")
                
                # Show processing results
                st.subheader("📊 Processing Results")
                
                for filename, result in processing_results.items():
                    if 'error' in result:
                        st.error(f"❌ {filename}: {result['error']}")
                    else:
                        summary = result['summary']
                        visual_contents = result['visual_contents']
                        
                        with st.expander(f"📄 {filename} - {summary['total_visuals']} visuals extracted"):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Visuals", summary['total_visuals'])
                            with col2:
                                st.metric("Avg Quality", f"{summary['avg_quality']:.2f}")
                            with col3:
                                st.metric("Processing Time", f"{summary['total_processing_time']:.2f}s")
                            with col4:
                                content_types = summary['content_types']
                                st.write(f"**Types**: {', '.join(content_types.keys())}")
                            
                            # Show sample visuals
                            if visual_contents:
                                st.write("**Sample Visual Content:**")
                                for i, visual in enumerate(visual_contents[:3]):  # Show first 3
                                    st.write(f"- {visual.content_type.title()}: Quality {visual.image_quality_score:.2f}, Confidence {visual.extraction_confidence:.2f}")
                                    if visual.financial_metrics:
                                        st.write(f"  Financial Metrics: {', '.join(visual.financial_metrics)}")
                                    if visual.time_periods:
                                        st.write(f"  Time Periods: {', '.join(visual.time_periods)}")
    
    # Test options for synthetic content
    st.header("🧪 Test with Synthetic Visual Content")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Test Chart Processing"):
            with st.spinner("Processing sample chart..."):
                # Create sample chart
                chart_image = create_sample_chart()
                
                # Process with visual content processor
                bounding_box = {'x': 0.1, 'y': 0.1, 'width': 0.8, 'height': 0.8}
                ocr_text = "Revenue Growth 2020-2024\n2020: $100M\n2021: $150M\n2022: $200M\n2023: $250M\n2024: $300M\nYear\nRevenue"
                
                visual_content = processor.process_visual_content(
                    image=chart_image,
                    document_id="test_chart_001",
                    page_number=1,
                    bounding_box=bounding_box,
                    ocr_text=ocr_text
                )
                
                # Store the visual content
                storage.store_visual_content(visual_content)
                
                st.success("✅ Chart processed successfully!")
                
                # Display results
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.image(chart_image, caption="Sample Chart", use_container_width=True)
                
                with col_b:
                    st.write("**Processing Results:**")
                    st.write(f"- Content Type: {visual_content.content_type}")
                    st.write(f"- Chart Type: {visual_content.chart_type}")
                    st.write(f"- Quality Score: {visual_content.image_quality_score:.3f}")
                    st.write(f"- Confidence: {visual_content.extraction_confidence:.3f}")
                    st.write(f"- Financial Metrics: {', '.join(visual_content.financial_metrics)}")
                    st.write(f"- Time Periods: {', '.join(visual_content.time_periods)}")
    
    with col2:
        if st.button("📋 Test Table Processing"):
            with st.spinner("Processing sample table..."):
                # Create sample table
                table_image = create_sample_table()
                
                # Process with visual content processor
                bounding_box = {'x': 0.1, 'y': 0.1, 'width': 0.8, 'height': 0.8}
                ocr_text = "Quarterly Financial Results\nQuarter\tRevenue\tProfit\tGrowth\nQ1 2024\t$75M\t$15M\t12%\nQ2 2024\t$82M\t$18M\t15%\nQ3 2024\t$89M\t$21M\t18%\nQ4 2024\t$95M\t$24M\t20%"
                
                visual_content = processor.process_visual_content(
                    image=table_image,
                    document_id="test_table_001",
                    page_number=1,
                    bounding_box=bounding_box,
                    ocr_text=ocr_text
                )
                
                # Store the visual content
                storage.store_visual_content(visual_content)
                
                st.success("✅ Table processed successfully!")
                
                # Display results
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.image(table_image, caption="Sample Table", use_container_width=True)
                
                with col_b:
                    st.write("**Processing Results:**")
                    st.write(f"- Content Type: {visual_content.content_type}")
                    st.write(f"- Table Type: {visual_content.table_type}")
                    st.write(f"- Quality Score: {visual_content.image_quality_score:.3f}")
                    st.write(f"- Confidence: {visual_content.extraction_confidence:.3f}")
                    st.write(f"- Financial Metrics: {', '.join(visual_content.financial_metrics)}")
                    st.write(f"- Data Quality: {visual_content.extracted_data.data_quality_score:.3f}")
    
    with col3:
        if st.button("🥧 Test Pie Chart Processing"):
            with st.spinner("Processing sample pie chart..."):
                # Create sample pie chart
                pie_image = create_sample_pie_chart()
                
                # Process with visual content processor
                bounding_box = {'x': 0.1, 'y': 0.1, 'width': 0.8, 'height': 0.8}
                ocr_text = "Market Share Distribution\nProduct A: 40%\nProduct B: 30%\nProduct C: 20%\nOthers: 10%"
                
                visual_content = processor.process_visual_content(
                    image=pie_image,
                    document_id="test_pie_001",
                    page_number=1,
                    bounding_box=bounding_box,
                    ocr_text=ocr_text
                )
                
                # Store the visual content
                storage.store_visual_content(visual_content)
                
                st.success("✅ Pie chart processed successfully!")
                
                # Display results
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.image(pie_image, caption="Sample Pie Chart", use_container_width=True)
                
                with col_b:
                    st.write("**Processing Results:**")
                    st.write(f"- Content Type: {visual_content.content_type}")
                    st.write(f"- Chart Type: {visual_content.chart_type}")
                    st.write(f"- Quality Score: {visual_content.image_quality_score:.3f}")
                    st.write(f"- Confidence: {visual_content.extraction_confidence:.3f}")
                    st.write(f"- Embedding Dimensions: {visual_content.visual_embedding.embedding_dimension}")
                    st.write(f"- Similarity Hash: {visual_content.visual_embedding.similarity_hash[:16]}...")
    
    # Show visual analytics dashboard
    st.header("📊 Visual Analytics Dashboard")
    
    # Get current statistics
    stats = storage.get_visual_statistics()
    
    if stats['total_visuals'] > 0:
        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Visuals", stats['total_visuals'])
        
        with col2:
            charts = stats['content_types'].get('chart', 0)
            st.metric("Charts", charts)
        
        with col3:
            tables = stats['content_types'].get('table', 0)
            st.metric("Tables", tables)
        
        with col4:
            avg_quality = stats['quality_stats'].get('avg_quality', 0)
            st.metric("Avg Quality", f"{avg_quality:.2f}")
        
        # Render full dashboard
        dashboard.render_dashboard()
        
        # Test similarity search
        st.header("🔍 Similarity Search Test")
        
        all_visuals = storage.get_all_visual_content()
        if len(all_visuals) >= 2:
            
            # Select a visual for similarity search
            visual_options = {v.visual_id: f"{v.content_type} - {v.document_id}" for v in all_visuals}
            selected_id = st.selectbox("Select visual for similarity search:", list(visual_options.keys()), format_func=lambda x: visual_options[x])
            
            if st.button("🔍 Find Similar Visuals"):
                similar_visuals = storage.find_similar_visuals(selected_id, similarity_threshold=0.3)
                
                if similar_visuals:
                    st.write(f"Found {len(similar_visuals)} similar visuals:")
                    
                    for similar in similar_visuals:
                        st.write(f"- {similar['visual_id']}: {similar['content_type']} (Similarity: {similar['similarity_score']:.3f})")
                else:
                    st.info("No similar visuals found")
        
        # Test clustering
        st.header("🎯 Clustering Analysis Test")
        
        if len(all_visuals) >= 3:
            n_clusters = st.slider("Number of clusters:", 2, min(5, len(all_visuals)), 3)
            
            if st.button("🎯 Perform Clustering"):
                cluster_results = storage.get_embedding_clusters(n_clusters)
                
                if 'error' not in cluster_results:
                    st.success(f"Successfully clustered {cluster_results['total_visuals']} visuals into {n_clusters} clusters")
                    
                    for cluster_id, visuals in cluster_results['clusters'].items():
                        st.write(f"**Cluster {cluster_id}** ({len(visuals)} visuals):")
                        for visual in visuals:
                            st.write(f"  - {visual['visual_id']}: {visual['content_type']} (Quality: {visual['quality_score']:.2f})")
                else:
                    st.error(cluster_results['error'])
    
    else:
        st.info("No visual content processed yet. Use the test buttons above to generate sample visual content.")
    
    # Clear storage option
    st.header("🗑️ Storage Management")
    
    if st.button("🗑️ Clear All Visual Content"):
        storage.clear_storage()
        st.success("All visual content cleared!")
        st.rerun()

if __name__ == "__main__":
    main()