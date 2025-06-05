"""
Visual Analytics Dashboard for FinDocPipeline
Comprehensive dashboard for visual content analytics and exploration.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import io
from PIL import Image

# Import storage
from storage.visual_content_storage import VisualContentStorage
from schemas.visual_content_schema import VisualContentSchema

class VisualAnalyticsDashboard:
    """Dashboard for comprehensive visual content analytics"""
    
    def __init__(self):
        self.storage = VisualContentStorage()
    
    def render_dashboard(self) -> None:
        """Render comprehensive visual analytics dashboard"""
        
        st.header("🖼️ Visual Content Analytics")
        
        # Get all visual content
        all_visuals = self._get_all_visual_content()
        
        if not all_visuals:
            st.info("📊 No visual content processed yet. Upload and process documents to see analytics.")
            self._render_empty_state()
            return
        
        # Create tabs for different analytics views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔍 Similarity Analysis", 
            "📈 Quality Metrics", 
            "🖼️ Visual Gallery", 
            "📋 Data Export"
        ])
        
        with tab1:
            self._render_overview_analytics(all_visuals)
        
        with tab2:
            self._render_similarity_analysis(all_visuals)
        
        with tab3:
            self._render_quality_metrics(all_visuals)
        
        with tab4:
            self._render_visual_gallery(all_visuals)
        
        with tab5:
            self._render_data_export(all_visuals)
    
    def _get_all_visual_content(self) -> List[Dict]:
        """Get all visual content as dictionaries"""
        visuals = self.storage.get_all_visual_content()
        return [visual.dict() for visual in visuals]
    
    def _render_empty_state(self) -> None:
        """Render empty state with instructions"""
        
        st.markdown("""
        ### 🚀 Get Started with Visual Content Analysis
        
        To see visual analytics:
        1. **Upload Documents**: Use the main pipeline to upload PDF, DOCX, or Excel files
        2. **Process Content**: The system will automatically extract and analyze visual elements
        3. **Explore Analytics**: Return here to explore charts, tables, and image insights
        
        #### What You'll See:
        - 📊 **Content Distribution**: Types of visual elements found
        - 🔍 **Similarity Analysis**: Find visually similar charts and tables
        - 📈 **Quality Metrics**: Assessment of image quality and extraction confidence
        - 🖼️ **Visual Gallery**: Browse and explore all extracted visuals
        - 📋 **Data Export**: Export metadata and embeddings for further analysis
        """)
    
    def _render_overview_analytics(self, visuals: List[Dict]) -> None:
        """Render comprehensive overview analytics"""
        
        st.subheader("📊 Overview Analytics")
        
        # Key metrics
        self._render_key_metrics(visuals)
        
        # Content distribution
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_content_type_distribution(visuals)
        
        with col2:
            self._render_document_distribution(visuals)
        
        # Timeline analysis
        self._render_processing_timeline(visuals)
        
        # Financial context analysis
        self._render_financial_context_analysis(visuals)
    
    def _render_key_metrics(self, visuals: List[Dict]) -> None:
        """Render key metrics cards"""
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Visuals", len(visuals))
        
        with col2:
            charts = [v for v in visuals if v['content_type'] == 'chart']
            st.metric("Charts", len(charts))
        
        with col3:
            tables = [v for v in visuals if v['content_type'] == 'table']
            st.metric("Tables", len(tables))
        
        with col4:
            if visuals:
                avg_quality = sum(v['image_quality_score'] for v in visuals) / len(visuals)
                st.metric("Avg Quality", f"{avg_quality:.2f}")
            else:
                st.metric("Avg Quality", "0.00")
        
        with col5:
            unique_docs = len(set(v['document_id'] for v in visuals))
            st.metric("Documents", unique_docs)
    
    def _render_content_type_distribution(self, visuals: List[Dict]) -> None:
        """Render content type distribution chart"""
        
        st.subheader("📈 Content Type Distribution")
        
        # Count by content type
        type_counts = {}
        for visual in visuals:
            content_type = visual['content_type'].title()
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        if type_counts:
            # Create pie chart
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="Visual Content Types",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No content type data available")
    
    def _render_document_distribution(self, visuals: List[Dict]) -> None:
        """Render document distribution chart"""
        
        st.subheader("📄 Document Distribution")
        
        # Count by document
        doc_counts = {}
        for visual in visuals:
            doc_id = visual['document_id']
            # Truncate long document IDs for display
            display_id = doc_id[:20] + "..." if len(doc_id) > 20 else doc_id
            doc_counts[display_id] = doc_counts.get(display_id, 0) + 1
        
        if doc_counts:
            # Create bar chart
            fig = px.bar(
                x=list(doc_counts.keys()),
                y=list(doc_counts.values()),
                title="Visuals per Document",
                labels={'x': 'Document', 'y': 'Visual Count'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No document distribution data available")
    
    def _render_processing_timeline(self, visuals: List[Dict]) -> None:
        """Render processing timeline"""
        
        st.subheader("⏱️ Processing Timeline")
        
        # Extract processing times and dates
        processing_data = []
        for visual in visuals:
            if 'processed_at' in visual and visual['processed_at']:
                try:
                    processed_at = datetime.fromisoformat(visual['processed_at'].replace('Z', '+00:00'))
                    processing_data.append({
                        'timestamp': processed_at,
                        'processing_time': visual.get('processing_time', 0),
                        'content_type': visual['content_type'],
                        'quality_score': visual['image_quality_score']
                    })
                except:
                    continue
        
        if processing_data:
            df = pd.DataFrame(processing_data)
            
            # Processing time over time
            fig = px.scatter(
                df, 
                x='timestamp', 
                y='processing_time',
                color='content_type',
                size='quality_score',
                title="Processing Time Timeline",
                labels={'processing_time': 'Processing Time (s)', 'timestamp': 'Processed At'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available")
    
    def _render_financial_context_analysis(self, visuals: List[Dict]) -> None:
        """Render financial context analysis"""
        
        st.subheader("💰 Financial Context Analysis")
        
        # Collect financial metrics
        all_metrics = []
        all_periods = []
        all_currencies = []
        
        for visual in visuals:
            all_metrics.extend(visual.get('financial_metrics', []))
            all_periods.extend(visual.get('time_periods', []))
            all_currencies.extend(visual.get('currencies', []))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if all_metrics:
                metric_counts = pd.Series(all_metrics).value_counts().head(10)
                fig = px.bar(
                    x=metric_counts.values,
                    y=metric_counts.index,
                    orientation='h',
                    title="Top Financial Metrics",
                    labels={'x': 'Frequency', 'y': 'Metric'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No financial metrics found")
        
        with col2:
            if all_periods:
                period_counts = pd.Series(all_periods).value_counts().head(10)
                fig = px.bar(
                    x=period_counts.values,
                    y=period_counts.index,
                    orientation='h',
                    title="Time Periods",
                    labels={'x': 'Frequency', 'y': 'Period'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time periods found")
        
        with col3:
            if all_currencies:
                currency_counts = pd.Series(all_currencies).value_counts()
                fig = px.pie(
                    values=currency_counts.values,
                    names=currency_counts.index,
                    title="Currencies Mentioned"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No currencies found")
    
    def _render_similarity_analysis(self, visuals: List[Dict]) -> None:
        """Render visual similarity analysis"""
        
        st.subheader("🔍 Visual Similarity Analysis")
        
        if len(visuals) < 2:
            st.info("Need at least 2 visuals for similarity analysis")
            return
        
        # Similarity search interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Search Parameters")
            
            # Select visual for similarity search
            visual_options = {
                v['visual_id']: f"{v['content_type'].title()} - {v['document_id'][:20]}..." 
                for v in visuals
            }
            
            selected_visual_id = st.selectbox(
                "Select visual for similarity search:",
                list(visual_options.keys()),
                format_func=lambda x: visual_options[x]
            )
            
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
            
            max_results = st.slider(
                "Max Results",
                min_value=1,
                max_value=20,
                value=5
            )
        
        with col2:
            if selected_visual_id:
                st.subheader("Similarity Results")
                
                # Find similar visuals
                similar_visuals = self.storage.find_similar_visuals(
                    selected_visual_id, 
                    similarity_threshold=similarity_threshold,
                    max_results=max_results
                )
                
                if similar_visuals:
                    st.write(f"Found {len(similar_visuals)} similar visuals:")
                    
                    # Display results
                    for i, similar in enumerate(similar_visuals):
                        with st.expander(f"Similar Visual {i+1} (Score: {similar['similarity_score']:.3f})"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write(f"**Visual ID**: {similar['visual_id']}")
                                st.write(f"**Content Type**: {similar['content_type']}")
                                st.write(f"**Document**: {similar['document_id']}")
                                st.write(f"**Similarity**: {similar['similarity_score']:.3f}")
                            
                            with col_b:
                                # Display visual if available
                                visual_data = self.storage.get_visual_content(similar['visual_id'])
                                if visual_data and visual_data.image_data:
                                    try:
                                        image_bytes = base64.b64decode(visual_data.image_data)
                                        st.image(image_bytes, caption="Similar Visual", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying image: {str(e)}")
                else:
                    st.info("No similar visuals found with the current threshold")
        
        # Clustering analysis
        st.subheader("🎯 Clustering Analysis")
        
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        
        if st.button("Perform Clustering Analysis"):
            cluster_results = self.storage.get_embedding_clusters(n_clusters=n_clusters)
            
            if 'error' in cluster_results:
                st.error(cluster_results['error'])
            else:
                st.success(f"Successfully clustered {cluster_results['total_visuals']} visuals into {n_clusters} clusters")
                
                # Display cluster information
                for cluster_id, cluster_visuals in cluster_results['clusters'].items():
                    with st.expander(f"Cluster {cluster_id} ({len(cluster_visuals)} visuals)"):
                        for visual in cluster_visuals:
                            st.write(f"- {visual['content_type']} from {visual['document_id']} (Quality: {visual['quality_score']:.2f})")
    
    def _render_quality_metrics(self, visuals: List[Dict]) -> None:
        """Render quality metrics analysis"""
        
        st.subheader("📈 Quality Metrics Analysis")
        
        # Quality distribution
        quality_scores = [v['image_quality_score'] for v in visuals]
        confidence_scores = [v['extraction_confidence'] for v in visuals]
        processing_times = [v['processing_time'] for v in visuals]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality score distribution
            fig = px.histogram(
                x=quality_scores,
                nbins=20,
                title="Image Quality Score Distribution",
                labels={'x': 'Quality Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence score distribution
            fig = px.histogram(
                x=confidence_scores,
                nbins=20,
                title="Extraction Confidence Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Quality vs Confidence scatter plot
        fig = px.scatter(
            x=quality_scores,
            y=confidence_scores,
            title="Quality vs Confidence Correlation",
            labels={'x': 'Image Quality Score', 'y': 'Extraction Confidence'},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Processing time analysis
        if processing_times:
            st.subheader("⏱️ Processing Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Processing Time", f"{np.mean(processing_times):.2f}s")
            
            with col2:
                st.metric("Min Processing Time", f"{np.min(processing_times):.2f}s")
            
            with col3:
                st.metric("Max Processing Time", f"{np.max(processing_times):.2f}s")
            
            # Processing time by content type
            processing_by_type = {}
            for visual in visuals:
                content_type = visual['content_type']
                if content_type not in processing_by_type:
                    processing_by_type[content_type] = []
                processing_by_type[content_type].append(visual['processing_time'])
            
            if processing_by_type:
                fig = px.box(
                    y=[time for times in processing_by_type.values() for time in times],
                    x=[content_type for content_type, times in processing_by_type.items() for _ in times],
                    title="Processing Time by Content Type",
                    labels={'x': 'Content Type', 'y': 'Processing Time (s)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_visual_gallery(self, visuals: List[Dict]) -> None:
        """Render visual content gallery"""
        
        st.subheader("🖼️ Visual Content Gallery")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            content_types = list(set(v['content_type'] for v in visuals))
            selected_type = st.selectbox("Filter by type:", ["All"] + content_types)
        
        with col2:
            documents = list(set(v['document_id'] for v in visuals))
            display_docs = [doc[:30] + "..." if len(doc) > 30 else doc for doc in documents]
            selected_doc_idx = st.selectbox("Filter by document:", ["All"] + list(range(len(documents))), 
                                          format_func=lambda x: "All" if x == "All" else display_docs[x])
        
        with col3:
            min_quality = st.slider("Minimum Quality Score", 0.0, 1.0, 0.0, 0.1)
        
        # Filter visuals
        filtered_visuals = visuals
        
        if selected_type != "All":
            filtered_visuals = [v for v in filtered_visuals if v['content_type'] == selected_type]
        
        if selected_doc_idx != "All":
            selected_doc = documents[selected_doc_idx]
            filtered_visuals = [v for v in filtered_visuals if v['document_id'] == selected_doc]
        
        filtered_visuals = [v for v in filtered_visuals if v['image_quality_score'] >= min_quality]
        
        st.write(f"Showing {len(filtered_visuals)} of {len(visuals)} visuals")
        
        # Display visuals in grid
        if filtered_visuals:
            cols_per_row = 3
            for i in range(0, len(filtered_visuals), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, visual in enumerate(filtered_visuals[i:i+cols_per_row]):
                    with cols[j]:
                        self._render_visual_card(visual)
        else:
            st.info("No visuals match the current filters")
    
    def _render_visual_card(self, visual: Dict) -> None:
        """Render individual visual card"""
        
        with st.container():
            st.write(f"**{visual['content_type'].title()}**")
            
            # Display image if available
            if visual.get('image_data'):
                try:
                    image_bytes = base64.b64decode(visual['image_data'])
                    st.image(image_bytes, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
            
            # Metadata
            st.write(f"📄 Document: {visual['document_id'][:20]}...")
            st.write(f"📊 Quality: {visual['image_quality_score']:.2f}")
            st.write(f"🎯 Confidence: {visual['extraction_confidence']:.2f}")
            
            if visual.get('title'):
                st.write(f"📝 Title: {visual['title'][:30]}...")
            
            # Show extracted data preview
            extracted_data = visual.get('extracted_data', {})
            if extracted_data and extracted_data.get('structured_data'):
                structured = extracted_data['structured_data']
                if visual['content_type'] == 'table' and 'headers' in structured:
                    st.write(f"📋 Table: {structured.get('num_rows', 0)}×{structured.get('num_columns', 0)}")
                    if 'rows' in structured and structured['rows']:
                        # Show first row as preview
                        first_row = structured['rows'][0] if structured['rows'] else []
                        if len(first_row) > 0:
                            preview = ', '.join(str(cell)[:10] for cell in first_row[:3])
                            st.write(f"📊 Data: {preview}...")
                elif visual['content_type'] == 'chart' and 'values' in structured:
                    st.write(f"📈 Points: {structured.get('data_points_count', 0)}")
                    st.write(f"📊 Range: {structured.get('min_value', 0):.1f}-{structured.get('max_value', 0):.1f}")
            
            # Details button
            if st.button(f"View Details", key=f"details_{visual['visual_id']}"):
                self._show_visual_details(visual)
    
    def _show_visual_details(self, visual: Dict) -> None:
        """Show detailed visual information in modal"""
        
        with st.expander(f"Visual Details - {visual['visual_id']}", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"- Visual ID: {visual['visual_id']}")
                st.write(f"- Document ID: {visual['document_id']}")
                st.write(f"- Content Type: {visual['content_type']}")
                st.write(f"- Page Number: {visual.get('page_number', 'N/A')}")
                
                st.write("**Quality Metrics:**")
                st.write(f"- Image Quality: {visual['image_quality_score']:.3f}")
                st.write(f"- Extraction Confidence: {visual['extraction_confidence']:.3f}")
                st.write(f"- Processing Time: {visual['processing_time']:.3f}s")
            
            with col2:
                st.write("**Content Analysis:**")
                if visual.get('title'):
                    st.write(f"- Title: {visual['title']}")
                if visual.get('financial_metrics'):
                    st.write(f"- Financial Metrics: {', '.join(visual['financial_metrics'])}")
                if visual.get('time_periods'):
                    st.write(f"- Time Periods: {', '.join(visual['time_periods'])}")
                if visual.get('currencies'):
                    st.write(f"- Currencies: {', '.join(visual['currencies'])}")
                
                # Display extracted data
                st.write("**Extracted Data:**")
                extracted_data = visual.get('extracted_data', {})
                if extracted_data:
                    if extracted_data.get('structured_data'):
                        structured = extracted_data['structured_data']
                        if visual['content_type'] == 'table' and 'headers' in structured:
                            st.write(f"- Headers: {', '.join(structured['headers'])}")
                            st.write(f"- Rows: {structured.get('num_rows', 0)}")
                            st.write(f"- Columns: {structured.get('num_columns', 0)}")
                            
                            # Display table data
                            if 'rows' in structured and structured['rows']:
                                st.write("**Table Data:**")
                                import pandas as pd
                                try:
                                    df = pd.DataFrame(structured['rows'], columns=structured['headers'])
                                    st.dataframe(df, use_container_width=True)
                                except Exception as e:
                                    st.write(f"Raw data: {structured['rows'][:3]}...")  # Show first 3 rows
                        
                        elif visual['content_type'] == 'chart' and 'values' in structured:
                            st.write(f"- Data Points: {structured.get('data_points_count', 0)}")
                            st.write(f"- Value Range: {structured.get('min_value', 0):.2f} - {structured.get('max_value', 0):.2f}")
                            st.write(f"- Mean Value: {structured.get('mean_value', 0):.2f}")
                    
                    st.write(f"- Data Quality: {extracted_data.get('data_quality_score', 0):.3f}")
                else:
                    st.write("- No structured data extracted")
                
                st.write("**Embedding Information:**")
                embedding = visual.get('visual_embedding', {})
                st.write(f"- Model: {embedding.get('embedding_model', 'N/A')}")
                st.write(f"- Dimensions: {embedding.get('embedding_dimension', 'N/A')}")
                st.write(f"- Similarity Hash: {embedding.get('similarity_hash', 'N/A')[:16]}...")
    
    def _render_data_export(self, visuals: List[Dict]) -> None:
        """Render data export interface"""
        
        st.subheader("📋 Data Export")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Options:**")
            
            export_metadata = st.checkbox("Include Metadata", value=True)
            export_embeddings = st.checkbox("Include Embeddings", value=False)
            export_images = st.checkbox("Include Image Data", value=False)
            
            export_format = st.selectbox("Export Format", ["JSON", "CSV"])
        
        with col2:
            st.write("**Statistics:**")
            stats = self.storage.get_visual_statistics()
            
            st.write(f"- Total Visuals: {stats['total_visuals']}")
            st.write(f"- Content Types: {len(stats['content_types'])}")
            st.write(f"- Documents: {len(stats['documents'])}")
            st.write(f"- Avg Quality: {stats['quality_stats'].get('avg_quality', 0):.3f}")
        
        # Export button
        if st.button("Generate Export"):
            export_data = self._generate_export_data(
                visuals, export_metadata, export_embeddings, export_images
            )
            
            if export_format == "JSON":
                self._download_json(export_data)
            else:
                self._download_csv(export_data)
    
    def _generate_export_data(
        self, 
        visuals: List[Dict], 
        include_metadata: bool,
        include_embeddings: bool,
        include_images: bool
    ) -> Dict[str, Any]:
        """Generate export data based on options"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_visuals': len(visuals),
            'visuals': []
        }
        
        for visual in visuals:
            visual_export = {
                'visual_id': visual['visual_id'],
                'document_id': visual['document_id'],
                'content_type': visual['content_type'],
                'image_quality_score': visual['image_quality_score'],
                'extraction_confidence': visual['extraction_confidence']
            }
            
            if include_metadata:
                visual_export.update({
                    'title': visual.get('title'),
                    'financial_metrics': visual.get('financial_metrics', []),
                    'time_periods': visual.get('time_periods', []),
                    'currencies': visual.get('currencies', []),
                    'processed_at': visual.get('processed_at')
                })
            
            if include_embeddings:
                visual_export['visual_embedding'] = visual.get('visual_embedding')
            
            if include_images:
                visual_export['image_data'] = visual.get('image_data')
            
            export_data['visuals'].append(visual_export)
        
        return export_data
    
    def _download_json(self, data: Dict[str, Any]) -> None:
        """Generate JSON download"""
        
        import json
        json_str = json.dumps(data, indent=2)
        
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"visual_content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _download_csv(self, data: Dict[str, Any]) -> None:
        """Generate CSV download"""
        
        # Flatten visual data for CSV
        flattened_data = []
        for visual in data['visuals']:
            flat_visual = {}
            for key, value in visual.items():
                if isinstance(value, list):
                    flat_visual[key] = '; '.join(map(str, value))
                elif isinstance(value, dict):
                    flat_visual[key] = str(value)
                else:
                    flat_visual[key] = value
            flattened_data.append(flat_visual)
        
        df = pd.DataFrame(flattened_data)
        csv_str = df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv_str,
            file_name=f"visual_content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )