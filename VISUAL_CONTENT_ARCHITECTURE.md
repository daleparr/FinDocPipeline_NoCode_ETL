# 🖼️ Visual Content & Image Embeddings Architecture

## 🎯 **Visual Content Processing Overview**

This architecture extends the multi-document processing system to handle visual elements (charts, graphs, tables, diagrams) with semantic embeddings and rich metadata storage, optimized for Streamlit deployment.

---

## 📊 **Visual Content Types in Financial Documents**

### **Chart Types**
- **Line Charts**: Time series data, trend analysis
- **Bar Charts**: Comparative metrics, quarterly results
- **Pie Charts**: Market share, revenue breakdown
- **Scatter Plots**: Correlation analysis, risk-return
- **Waterfall Charts**: Cash flow, variance analysis
- **Heatmaps**: Performance matrices, correlation maps

### **Table Types**
- **Financial Statements**: Income statement, balance sheet
- **Data Tables**: Quarterly results, segment performance
- **Comparison Tables**: Peer analysis, historical data
- **Summary Tables**: Key metrics, ratios

### **Diagram Types**
- **Organizational Charts**: Corporate structure
- **Process Flows**: Business operations
- **Geographic Maps**: Regional performance
- **Infographics**: Key highlights, summaries

---

## 🏗️ **Visual Content Schema Design**

### **Base Visual Content Schema**
```python
# File: schemas/visual_content_schema.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import numpy as np
from enum import Enum

class VisualContentType(str, Enum):
    CHART = "chart"
    TABLE = "table"
    DIAGRAM = "diagram"
    INFOGRAPHIC = "infographic"
    IMAGE = "image"
    UNKNOWN = "unknown"

class ChartType(str, Enum):
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    WATERFALL = "waterfall"
    HEATMAP = "heatmap"
    COMBO_CHART = "combo_chart"
    UNKNOWN = "unknown"

class TableType(str, Enum):
    FINANCIAL_STATEMENT = "financial_statement"
    DATA_TABLE = "data_table"
    COMPARISON_TABLE = "comparison_table"
    SUMMARY_TABLE = "summary_table"
    UNKNOWN = "unknown"

class BoundingBox(BaseModel):
    """Bounding box coordinates for visual element"""
    x: float = Field(..., description="X coordinate (normalized 0-1)")
    y: float = Field(..., description="Y coordinate (normalized 0-1)")
    width: float = Field(..., description="Width (normalized 0-1)")
    height: float = Field(..., description="Height (normalized 0-1)")
    
    @validator('x', 'y', 'width', 'height')
    def validate_coordinates(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Coordinates must be normalized between 0 and 1')
        return v

class VisualEmbedding(BaseModel):
    """Visual content embedding representation"""
    embedding_vector: List[float] = Field(..., description="Image embedding vector")
    embedding_model: str = Field(..., description="Model used for embedding")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    similarity_hash: str = Field(..., description="Perceptual hash for similarity")
    
    @validator('embedding_vector')
    def validate_embedding_vector(cls, v):
        if len(v) == 0:
            raise ValueError('Embedding vector cannot be empty')
        return v

class ExtractedData(BaseModel):
    """Data extracted from visual content"""
    raw_data: Optional[List[List[Any]]] = Field(None, description="Raw extracted data")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Structured data representation")
    data_points: Optional[List[Dict[str, Any]]] = Field(None, description="Individual data points")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Data extraction quality")

class VisualContentSchema(BaseModel):
    """Comprehensive visual content schema"""
    
    # Core identification
    visual_id: str = Field(..., description="Unique visual content identifier")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: Optional[int] = Field(None, description="Page number where found")
    
    # Visual content classification
    content_type: VisualContentType = Field(..., description="Type of visual content")
    chart_type: Optional[ChartType] = Field(None, description="Specific chart type")
    table_type: Optional[TableType] = Field(None, description="Specific table type")
    
    # Spatial information
    bounding_box: BoundingBox = Field(..., description="Location within document")
    area_percentage: float = Field(..., ge=0.0, le=1.0, description="Percentage of page area")
    
    # Visual embeddings
    visual_embedding: VisualEmbedding = Field(..., description="Image embedding representation")
    
    # Content analysis
    title: Optional[str] = Field(None, description="Chart/table title")
    caption: Optional[str] = Field(None, description="Caption or description")
    axis_labels: Optional[List[str]] = Field(None, description="Axis labels for charts")
    legend_items: Optional[List[str]] = Field(None, description="Legend items")
    
    # Extracted data
    extracted_data: ExtractedData = Field(..., description="Data extracted from visual")
    
    # Financial context
    financial_metrics: List[str] = Field(default_factory=list, description="Financial metrics shown")
    time_periods: List[str] = Field(default_factory=list, description="Time periods covered")
    currencies: List[str] = Field(default_factory=list, description="Currencies mentioned")
    
    # Quality metrics
    image_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Image quality assessment")
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Data extraction confidence")
    
    # Processing metadata
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    
    # Raw image data (base64 encoded for storage)
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_format: Optional[str] = Field(None, description="Image format (PNG, JPEG, etc.)")
    image_size: Optional[Dict[str, int]] = Field(None, description="Image dimensions")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist()
        }
```

---

## 🔧 **Visual Content Processing Engine**

### **Streamlit-Optimized Visual Processor**
```python
# File: processing/visual_content_processor.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from schemas.visual_content_schema import VisualContentSchema, VisualEmbedding, BoundingBox

class StreamlitVisualProcessor:
    """Streamlit-optimized visual content processor"""
    
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.chart_classifier = self._load_chart_classifier()
        self.table_detector = self._load_table_detector()
    
    @st.cache_resource
    def _load_embedding_model(_self):
        """Load lightweight image embedding model"""
        # Use a lightweight approach instead of heavy models
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        return {
            'pca': PCA(n_components=128),  # Reduce to 128 dimensions
            'scaler': StandardScaler(),
            'model_name': 'lightweight_visual_embeddings'
        }
    
    def process_visual_content(
        self, 
        image: np.ndarray, 
        document_id: str, 
        page_number: int,
        bounding_box: Dict[str, float]
    ) -> VisualContentSchema:
        """Process visual content and generate comprehensive metadata"""
        
        # Generate unique visual ID
        visual_id = self._generate_visual_id(document_id, page_number, bounding_box)
        
        # Classify visual content type
        content_type, chart_type, table_type = self._classify_visual_content(image)
        
        # Generate visual embedding
        visual_embedding = self._generate_visual_embedding(image)
        
        # Extract data from visual content
        extracted_data = self._extract_data_from_visual(image, content_type)
        
        # Analyze content
        content_analysis = self._analyze_visual_content(image, content_type)
        
        # Assess quality
        quality_metrics = self._assess_visual_quality(image)
        
        # Convert image to base64 for storage
        image_data, image_format, image_size = self._encode_image(image)
        
        return VisualContentSchema(
            visual_id=visual_id,
            document_id=document_id,
            page_number=page_number,
            content_type=content_type,
            chart_type=chart_type,
            table_type=table_type,
            bounding_box=BoundingBox(**bounding_box),
            area_percentage=self._calculate_area_percentage(bounding_box),
            visual_embedding=visual_embedding,
            title=content_analysis.get('title'),
            caption=content_analysis.get('caption'),
            axis_labels=content_analysis.get('axis_labels', []),
            legend_items=content_analysis.get('legend_items', []),
            extracted_data=extracted_data,
            financial_metrics=content_analysis.get('financial_metrics', []),
            time_periods=content_analysis.get('time_periods', []),
            currencies=content_analysis.get('currencies', []),
            image_quality_score=quality_metrics['quality_score'],
            extraction_confidence=quality_metrics['extraction_confidence'],
            image_data=image_data,
            image_format=image_format,
            image_size=image_size
        )
    
    @st.cache_data
    def _generate_visual_embedding(_self, image: np.ndarray) -> VisualEmbedding:
        """Generate lightweight visual embedding"""
        
        # Resize image for consistent processing
        resized_image = cv2.resize(image, (224, 224))
        
        # Convert to grayscale for feature extraction
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Extract basic visual features
        features = _self._extract_visual_features(gray_image)
        
        # Generate perceptual hash for similarity
        similarity_hash = _self._generate_perceptual_hash(gray_image)
        
        return VisualEmbedding(
            embedding_vector=features.tolist(),
            embedding_model='lightweight_visual_features',
            embedding_dimension=len(features),
            similarity_hash=similarity_hash
        )
    
    def _extract_visual_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract lightweight visual features"""
        
        features = []
        
        # Histogram features
        hist = cv2.calcHist([gray_image], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Texture features (LBP-like)
        texture_features = self._extract_texture_features(gray_image)
        features.extend(texture_features)
        
        # Shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_features = [
            len(contours),  # Number of contours
            np.mean([cv2.contourArea(c) for c in contours]) if contours else 0,  # Average area
            np.std([cv2.contourArea(c) for c in contours]) if contours else 0   # Area variance
        ]
        features.extend(shape_features)
        
        return np.array(features)
    
    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract basic texture features"""
        
        # Calculate local binary pattern-like features
        rows, cols = image.shape
        features = []
        
        # Sample texture at different scales
        for scale in [1, 2, 4]:
            if rows > scale and cols > scale:
                sampled = image[::scale, ::scale]
                
                # Calculate variance (texture measure)
                variance = np.var(sampled)
                features.append(variance)
                
                # Calculate gradient magnitude
                grad_x = cv2.Sobel(sampled, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(sampled, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                features.append(gradient_magnitude)
        
        return features
    
    def _generate_perceptual_hash(self, image: np.ndarray) -> str:
        """Generate perceptual hash for image similarity"""
        
        # Resize to 8x8 for hash
        small_image = cv2.resize(image, (8, 8))
        
        # Calculate average
        avg = np.mean(small_image)
        
        # Create binary hash
        binary_hash = small_image > avg
        
        # Convert to hex string
        hash_string = ''.join(['1' if pixel else '0' for pixel in binary_hash.flatten()])
        
        # Convert binary to hex
        hex_hash = hex(int(hash_string, 2))[2:]
        
        return hex_hash
    
    def _classify_visual_content(self, image: np.ndarray) -> Tuple[str, Optional[str], Optional[str]]:
        """Classify visual content type"""
        
        # Basic classification using image analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines (charts often have many lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        # Detect rectangles (tables often have rectangular structure)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_count = sum(1 for c in contours if self._is_rectangular(c))
        
        # Classification logic
        if rect_count > 5 and line_count > 10:
            return "table", None, "data_table"
        elif line_count > 20:
            return "chart", "line_chart", None
        elif rect_count > 2:
            return "chart", "bar_chart", None
        else:
            return "image", None, None
    
    def _is_rectangular(self, contour) -> bool:
        """Check if contour is approximately rectangular"""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return len(approx) == 4
```

---

## 💾 **Visual Content Storage Architecture**

### **Streamlit Session State Storage**
```python
# File: storage/visual_content_storage.py
import streamlit as st
import json
import pickle
import base64
from typing import List, Dict, Any, Optional
from schemas.visual_content_schema import VisualContentSchema

class StreamlitVisualStorage:
    """Streamlit-optimized visual content storage"""
    
    def __init__(self):
        self.storage_key = "visual_content_storage"
        self.embedding_key = "visual_embeddings"
        self.similarity_index_key = "visual_similarity_index"
        
        # Initialize storage in session state
        if self.storage_key not in st.session_state:
            st.session_state[self.storage_key] = {}
        
        if self.embedding_key not in st.session_state:
            st.session_state[self.embedding_key] = {}
        
        if self.similarity_index_key not in st.session_state:
            st.session_state[self.similarity_index_key] = {}
    
    def store_visual_content(self, visual_content: VisualContentSchema) -> None:
        """Store visual content with embeddings"""
        
        # Store main content
        st.session_state[self.storage_key][visual_content.visual_id] = visual_content.dict()
        
        # Store embedding separately for efficient similarity search
        st.session_state[self.embedding_key][visual_content.visual_id] = {
            'embedding_vector': visual_content.visual_embedding.embedding_vector,
            'similarity_hash': visual_content.visual_embedding.similarity_hash,
            'content_type': visual_content.content_type,
            'document_id': visual_content.document_id
        }
        
        # Update similarity index
        self._update_similarity_index(visual_content)
    
    def get_visual_content(self, visual_id: str) -> Optional[VisualContentSchema]:
        """Retrieve visual content by ID"""
        
        content_data = st.session_state[self.storage_key].get(visual_id)
        if content_data:
            return VisualContentSchema(**content_data)
        return None
    
    def find_similar_visuals(
        self, 
        visual_id: str, 
        similarity_threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Find visually similar content"""
        
        target_embedding = st.session_state[self.embedding_key].get(visual_id)
        if not target_embedding:
            return []
        
        similar_visuals = []
        target_vector = np.array(target_embedding['embedding_vector'])
        
        for other_id, other_embedding in st.session_state[self.embedding_key].items():
            if other_id == visual_id:
                continue
            
            # Calculate cosine similarity
            other_vector = np.array(other_embedding['embedding_vector'])
            similarity = self._cosine_similarity(target_vector, other_vector)
            
            if similarity >= similarity_threshold:
                similar_visuals.append({
                    'visual_id': other_id,
                    'similarity_score': similarity,
                    'content_type': other_embedding['content_type'],
                    'document_id': other_embedding['document_id']
                })
        
        # Sort by similarity and return top results
        similar_visuals.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_visuals[:max_results]
    
    def get_visuals_by_document(self, document_id: str) -> List[VisualContentSchema]:
        """Get all visual content for a document"""
        
        visuals = []
        for visual_data in st.session_state[self.storage_key].values():
            if visual_data['document_id'] == document_id:
                visuals.append(VisualContentSchema(**visual_data))
        
        return visuals
    
    def get_visuals_by_type(self, content_type: str) -> List[VisualContentSchema]:
        """Get all visual content of a specific type"""
        
        visuals = []
        for visual_data in st.session_state[self.storage_key].values():
            if visual_data['content_type'] == content_type:
                visuals.append(VisualContentSchema(**visual_data))
        
        return visuals
    
    def export_visual_metadata(self) -> Dict[str, Any]:
        """Export all visual metadata for analysis"""
        
        export_data = {
            'visual_content': st.session_state[self.storage_key],
            'embeddings': st.session_state[self.embedding_key],
            'similarity_index': st.session_state[self.similarity_index_key],
            'export_timestamp': datetime.now().isoformat(),
            'total_visuals': len(st.session_state[self.storage_key])
        }
        
        return export_data
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
```

---

## 📊 **Visual Analytics Dashboard**

### **Visual Content Analytics Interface**
```python
# File: ui/visual_analytics_dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
from storage.visual_content_storage import StreamlitVisualStorage

class VisualAnalyticsDashboard:
    """Dashboard for visual content analytics"""
    
    def __init__(self):
        self.storage = StreamlitVisualStorage()
    
    def render_dashboard(self) -> None:
        """Render comprehensive visual analytics dashboard"""
        
        st.header("🖼️ Visual Content Analytics")
        
        # Get all visual content
        all_visuals = self._get_all_visual_content()
        
        if not all_visuals:
            st.info("No visual content processed yet. Upload and process documents to see analytics.")
            return
        
        # Overview metrics
        self._render_overview_metrics(all_visuals)
        
        # Content type distribution
        self._render_content_type_distribution(all_visuals)
        
        # Visual similarity analysis
        self._render_similarity_analysis(all_visuals)
        
        # Quality metrics
        self._render_quality_metrics(all_visuals)
        
        # Visual content gallery
        self._render_visual_gallery(all_visuals)
    
    def _render_overview_metrics(self, visuals: List[Dict]) -> None:
        """Render overview metrics"""
        
        st.subheader("📊 Overview Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Visuals", len(visuals))
        
        with col2:
            charts = [v for v in visuals if v['content_type'] == 'chart']
            st.metric("Charts", len(charts))
        
        with col3:
            tables = [v for v in visuals if v['content_type'] == 'table']
            st.metric("Tables", len(tables))
        
        with col4:
            avg_quality = sum(v['image_quality_score'] for v in visuals) / len(visuals)
            st.metric("Avg Quality", f"{avg_quality:.2f}")
    
    def _render_content_type_distribution(self, visuals: List[Dict]) -> None:
        """Render content type distribution"""
        
        st.subheader("📈 Content Type Distribution")
        
        # Count by content type
        type_counts = {}
        for visual in visuals:
            content_type = visual['content_type']
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        # Create pie chart
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Visual Content Types"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_similarity_analysis(self, visuals: List[Dict]) -> None:
        """Render visual similarity analysis"""
        
        st.subheader("🔍 Visual Similarity Analysis")
        
        if len(visuals) < 2:
            st.info("Need at least 2 visuals for similarity analysis")
            return
        
        # Select visual for similarity search
        visual_options = {v['visual_id']: f"{v['content_type']} - {v['document_id']}" for v in visuals}
        selected_visual_id = st.selectbox("Select visual for similarity search:", list(visual_options.keys()), format_func=lambda x: visual_options[x])
        
        if selected_visual_id:
            # Find similar visuals
            similar_visuals = self.storage.find_similar_visuals(selected_visual_id, similarity_threshold=0.5)
            
            if similar_visuals:
                st.write(f"Found {len(similar_visuals)} similar visuals:")
                
                for similar in similar_visuals:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"Visual ID: {similar['visual_id']}")
                    
                    with col2:
                        st.write(f"Similarity: {similar['similarity_score']:.3f}")
                    
                    with col3:
                        st.write(f"Type: {similar['content_type']}")
            else:
                st.info("No similar visuals found")
    
    def _render_visual_gallery(self, visuals: List[Dict]) -> None:
        """Render visual content gallery"""
        
        st.subheader("🖼️ Visual Content Gallery")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            content_types = list(set(v['content_type'] for v in visuals))
            selected_type = st.selectbox("Filter by type:", ["All"] + content_types)
        
        with col2:
            documents = list(set(v['document_id'] for v in visuals))
            selected_doc = st.selectbox("Filter by document:", ["All"] + documents)
        
        # Filter visuals
        filtered_visuals = visuals
        if selected_type != "All":
            filtered_visuals = [v for v in filtered_visuals if v['content_type'] == selected_type]
        if selected_doc != "All":
            filtered_visuals = [v for v in filtered_visuals if v['document_id'] == selected_doc]
        
        # Display visuals in grid
        cols_per_row = 3
        for i in range(0, len(filtered_visuals), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, visual in enumerate(filtered_visuals[i:i+cols_per_row]):
                with cols[j]:
                    st.write(f"**{visual['content_type'].title()}**")
                    st.write(f"Document: {visual['document_id']}")
                    st.write(f"Quality: {visual['image_quality_score']:.2f}")
                    
                    # Display image if available
                    if visual.get('image_data'):
                        try:
                            image_bytes = base64.b64decode(visual['image_data'])
                            st.image(image_bytes, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
                    
                    if st.button(f"View Details", key=f"details_{visual['visual_id']}"):
                        self._show_visual_details(visual)
    
    def _show_visual_details(self, visual: Dict) -> None:
        """Show detailed visual information"""
        
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
            
            with col2:
                st.write("**Content Analysis:**")
                if visual.get('title'):
                    st.write(f"- Title: {visual['title']}")
                if visual.get('financial_metrics'):
                    st.write(f"- Financial Metrics: {', '.join(visual['financial_metrics'])}")
                if visual.get('time_periods'):
                    st.write(f"- Time Periods: {', '.join(visual['time_periods'])}")
                
                st.write("**Embedding Information:**")
                embedding = visual.get('visual_embedding', {})
                st.write(f"- Model: {embedding.get('embedding_model', 'N/A')}")
                st.write(f"- Dimensions: {embedding.get('embedding_dimension', 'N/A')}")
                st.write(f"- Similarity Hash: {embedding.get('similarity_hash', 'N/A')[:16]}...")
```

This visual content architecture provides comprehensive image processing, embedding generation, and metadata storage capabilities while remaining compatible with Streamlit's constraints. The system can handle charts, tables, and other visual elements with rich semantic understanding and similarity search capabilities.