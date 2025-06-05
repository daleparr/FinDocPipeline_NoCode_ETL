"""
Visual Content Schema for FinDocPipeline
Comprehensive schema for visual content processing, embeddings, and metadata.
"""

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
    """Comprehensive visual content schema with rich relational context"""
    
    # Core identification
    visual_id: str = Field(..., description="Unique visual content identifier")
    document_id: str = Field(..., description="Parent document identifier")
    page_number: Optional[int] = Field(None, description="Page number where found")
    
    # Document context (NEW: Enhanced relational context)
    document_name: Optional[str] = Field(None, description="Original document filename")
    document_type: Optional[str] = Field(None, description="Type of document (annual report, quarterly, etc.)")
    bank_name: Optional[str] = Field(None, description="Bank or institution name extracted from document")
    reporting_period: Optional[str] = Field(None, description="Reporting period (Q1 2024, FY 2023, etc.)")
    fiscal_year: Optional[str] = Field(None, description="Fiscal year")
    quarter: Optional[str] = Field(None, description="Quarter (Q1, Q2, Q3, Q4)")
    
    # Visual content classification
    content_type: VisualContentType = Field(..., description="Type of visual content")
    chart_type: Optional[ChartType] = Field(None, description="Specific chart type")
    table_type: Optional[TableType] = Field(None, description="Specific table type")
    
    # Spatial information
    bounding_box: BoundingBox = Field(..., description="Location within document")
    area_percentage: float = Field(..., ge=0.0, le=1.0, description="Percentage of page area")
    
    # Visual embeddings
    visual_embedding: VisualEmbedding = Field(..., description="Image embedding representation")
    
    # Enhanced content analysis (NEW: Rich contextual metadata)
    title: Optional[str] = Field(None, description="Chart/table main title")
    subtitle: Optional[str] = Field(None, description="Chart/table subtitle")
    caption: Optional[str] = Field(None, description="Caption or description")
    
    # Chart-specific context
    x_axis_title: Optional[str] = Field(None, description="X-axis title/label")
    y_axis_title: Optional[str] = Field(None, description="Y-axis title/label")
    axis_labels: Optional[List[str]] = Field(None, description="All axis labels for charts")
    legend_items: Optional[List[str]] = Field(None, description="Legend items with descriptions")
    legend_title: Optional[str] = Field(None, description="Legend title")
    
    # Table-specific context
    table_headers: Optional[List[str]] = Field(None, description="Table column headers")
    table_title: Optional[str] = Field(None, description="Table title")
    row_labels: Optional[List[str]] = Field(None, description="Table row labels")
    
    # Extracted data
    extracted_data: ExtractedData = Field(..., description="Data extracted from visual")
    
    # Enhanced financial context (NEW: More granular financial information)
    financial_metrics: List[str] = Field(default_factory=list, description="Financial metrics shown")
    time_periods: List[str] = Field(default_factory=list, description="Time periods covered")
    currencies: List[str] = Field(default_factory=list, description="Currencies mentioned")
    business_segments: List[str] = Field(default_factory=list, description="Business segments mentioned")
    geographic_regions: List[str] = Field(default_factory=list, description="Geographic regions mentioned")
    
    # Relational context (NEW: Cross-reference capabilities)
    related_visuals: List[str] = Field(default_factory=list, description="IDs of related visual content")
    section_context: Optional[str] = Field(None, description="Document section where visual appears")
    page_context: Optional[str] = Field(None, description="Context of the page content")
    
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