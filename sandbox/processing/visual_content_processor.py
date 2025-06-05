"""
Visual Content Processor for FinDocPipeline
Handles image/chart/graph/table embeddings with metadata extraction.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import hashlib
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import schemas
from schemas.visual_content_schema import (
    VisualContentSchema, VisualEmbedding, BoundingBox, ExtractedData,
    VisualContentType, ChartType, TableType
)

# Import ML libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

class VisualContentProcessor:
    """
    Comprehensive visual content processor for financial documents.
    Handles charts, graphs, tables, and images with semantic embeddings.
    """
    
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.chart_classifier = self._load_chart_classifier()
        self.table_detector = self._load_table_detector()
        self.financial_terms = self._load_financial_terms()
        
    @st.cache_resource
    def _load_embedding_model(_self):
        """Load lightweight image embedding model"""
        return {
            'pca': PCA(n_components=128),  # Reduce to 128 dimensions
            'scaler': StandardScaler(),
            'model_name': 'lightweight_visual_embeddings',
            'initialized': False
        }
    
    @st.cache_resource
    def _load_chart_classifier(_self):
        """Load chart classification model"""
        return {
            'kmeans': KMeans(n_clusters=6, random_state=42),  # 6 chart types
            'feature_weights': {
                'line_density': 0.3,
                'rect_count': 0.25,
                'circle_count': 0.2,
                'edge_density': 0.15,
                'color_variance': 0.1
            }
        }
    
    @st.cache_resource
    def _load_table_detector(_self):
        """Load table detection model"""
        return {
            'min_rows': 2,
            'min_cols': 2,
            'line_threshold': 0.7,
            'cell_aspect_ratio': (0.5, 3.0)
        }
    
    def _load_financial_terms(self) -> Dict[str, List[str]]:
        """Load financial terms for context extraction"""
        return {
            'metrics': [
                'revenue', 'profit', 'loss', 'ebitda', 'margin', 'roi', 'eps',
                'cash flow', 'assets', 'liabilities', 'equity', 'debt',
                'growth', 'return', 'yield', 'ratio', 'percentage', '%'
            ],
            'time_periods': [
                'q1', 'q2', 'q3', 'q4', 'quarter', 'annual', 'yearly',
                'monthly', 'weekly', 'daily', 'ytd', 'mtd', 'qtd',
                '2020', '2021', '2022', '2023', '2024', '2025'
            ],
            'currencies': [
                'usd', 'eur', 'gbp', 'jpy', 'cad', 'aud', 'chf',
                '$', '€', '£', '¥', 'dollar', 'euro', 'pound'
            ],
            'banks': [
                'jpmorgan', 'chase', 'bank of america', 'wells fargo', 'citigroup',
                'goldman sachs', 'morgan stanley', 'hsbc', 'barclays', 'deutsche bank',
                'ubs', 'credit suisse', 'bnp paribas', 'santander', 'ing',
                'royal bank', 'td bank', 'scotiabank', 'bmo', 'rbc'
            ],
            'business_segments': [
                'retail banking', 'commercial banking', 'investment banking',
                'wealth management', 'asset management', 'trading', 'markets',
                'corporate banking', 'consumer banking', 'private banking'
            ],
            'geographic_regions': [
                'north america', 'europe', 'asia pacific', 'latin america',
                'middle east', 'africa', 'usa', 'canada', 'uk', 'germany',
                'france', 'japan', 'china', 'australia', 'brazil'
            ]
        }
    
    def process_visual_content(
        self,
        image: np.ndarray,
        document_id: str,
        page_number: int,
        bounding_box: Dict[str, float],
        ocr_text: Optional[str] = None,
        document_name: Optional[str] = None,
        page_context: Optional[str] = None
    ) -> VisualContentSchema:
        """
        Process visual content and generate comprehensive metadata.
        
        Args:
            image: Image as numpy array
            document_id: Parent document identifier
            page_number: Page number where found
            bounding_box: Location within document
            ocr_text: Optional OCR text from the visual
            
        Returns:
            VisualContentSchema with complete analysis
        """
        start_time = time.time()
        
        # Generate unique visual ID
        visual_id = self._generate_visual_id(document_id, page_number, bounding_box)
        
        # Classify visual content type
        content_type, chart_type, table_type = self._classify_visual_content(image)
        
        # Generate visual embedding
        visual_embedding = self._generate_visual_embedding(image)
        
        # Extract data from visual content
        extracted_data = self._extract_data_from_visual(image, content_type, ocr_text)
        
        # Enhanced content analysis with relational context
        content_analysis = self._analyze_visual_content(image, content_type, ocr_text)
        
        # Extract document context
        document_context = self._extract_document_context(document_name, ocr_text, page_context)
        
        # Extract enhanced visual metadata
        visual_metadata = self._extract_enhanced_visual_metadata(content_type, ocr_text)
        
        # Assess quality metrics
        quality_metrics = self._assess_visual_quality(image)
        
        # Convert image to base64 for storage
        image_data, image_format, image_size = self._encode_image(image)
        
        processing_time = time.time() - start_time
        
        return VisualContentSchema(
            visual_id=visual_id,
            document_id=document_id,
            page_number=page_number,
            
            # Enhanced document context
            document_name=document_context.get('document_name'),
            document_type=document_context.get('document_type'),
            bank_name=document_context.get('bank_name'),
            reporting_period=document_context.get('reporting_period'),
            fiscal_year=document_context.get('fiscal_year'),
            quarter=document_context.get('quarter'),
            
            content_type=content_type,
            chart_type=chart_type,
            table_type=table_type,
            bounding_box=BoundingBox(**bounding_box),
            area_percentage=self._calculate_area_percentage(bounding_box),
            visual_embedding=visual_embedding,
            
            # Enhanced visual metadata
            title=visual_metadata.get('title'),
            subtitle=visual_metadata.get('subtitle'),
            caption=visual_metadata.get('caption'),
            x_axis_title=visual_metadata.get('x_axis_title'),
            y_axis_title=visual_metadata.get('y_axis_title'),
            axis_labels=visual_metadata.get('axis_labels', []),
            legend_items=visual_metadata.get('legend_items', []),
            legend_title=visual_metadata.get('legend_title'),
            table_headers=visual_metadata.get('table_headers', []),
            table_title=visual_metadata.get('table_title'),
            row_labels=visual_metadata.get('row_labels', []),
            
            extracted_data=extracted_data,
            
            # Enhanced financial context
            financial_metrics=content_analysis.get('financial_metrics', []),
            time_periods=content_analysis.get('time_periods', []),
            currencies=content_analysis.get('currencies', []),
            business_segments=content_analysis.get('business_segments', []),
            geographic_regions=content_analysis.get('geographic_regions', []),
            
            # Relational context
            section_context=document_context.get('section_context'),
            page_context=page_context,
            
            image_quality_score=quality_metrics['quality_score'],
            extraction_confidence=quality_metrics['extraction_confidence'],
            processing_time=processing_time,
            image_data=image_data,
            image_format=image_format,
            image_size=image_size
        )
    
    def _generate_visual_id(self, document_id: str, page_number: int, bounding_box: Dict[str, float]) -> str:
        """Generate unique visual content identifier"""
        content = f"{document_id}_{page_number}_{bounding_box['x']:.3f}_{bounding_box['y']:.3f}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_visual_embedding(self, image: np.ndarray) -> VisualEmbedding:
        """Generate comprehensive visual embedding"""
        
        # Resize image for consistent processing
        resized_image = cv2.resize(image, (224, 224))
        
        # Convert to grayscale for feature extraction
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Extract comprehensive visual features
        features = self._extract_visual_features(gray_image, resized_image)
        
        # Generate perceptual hash for similarity
        similarity_hash = self._generate_perceptual_hash(gray_image)
        
        return VisualEmbedding(
            embedding_vector=features.tolist(),
            embedding_model='comprehensive_visual_features',
            embedding_dimension=len(features),
            similarity_hash=similarity_hash
        )
    
    def _extract_visual_features(self, gray_image: np.ndarray, color_image: np.ndarray) -> np.ndarray:
        """Extract comprehensive visual features for embedding"""
        
        features = []
        
        # 1. Histogram features (color and intensity)
        gray_hist = cv2.calcHist([gray_image], [0], None, [32], [0, 256])
        features.extend(gray_hist.flatten())
        
        # Color histograms for each channel
        for i in range(3):
            color_hist = cv2.calcHist([color_image], [i], None, [16], [0, 256])
            features.extend(color_hist.flatten())
        
        # 2. Edge and contour features
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Line detection (important for charts)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        features.append(line_count / 100.0)  # Normalize
        
        # 3. Shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Contour statistics
        features.append(len(contours) / 50.0)  # Normalize contour count
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features.extend([
                np.mean(areas) / 10000.0,  # Average area
                np.std(areas) / 10000.0,   # Area variance
                max(areas) / 10000.0       # Max area
            ])
            
            # Shape complexity
            perimeters = [cv2.arcLength(c, True) for c in contours]
            if perimeters:
                features.append(np.mean(perimeters) / 1000.0)
        else:
            features.extend([0, 0, 0, 0])
        
        # 4. Texture features
        texture_features = self._extract_texture_features(gray_image)
        features.extend(texture_features)
        
        # 5. Geometric features (rectangles, circles)
        rect_count = sum(1 for c in contours if self._is_rectangular(c))
        circle_count = self._count_circles(gray_image)
        features.extend([rect_count / 20.0, circle_count / 10.0])
        
        # 6. Color variance and distribution
        color_variance = np.var(color_image.reshape(-1, 3), axis=0)
        features.extend(color_variance / 10000.0)
        
        # 7. Spatial frequency features
        fft_features = self._extract_frequency_features(gray_image)
        features.extend(fft_features)
        
        return np.array(features)
    
    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features using multiple scales"""
        
        features = []
        
        # Multi-scale texture analysis
        for scale in [1, 2, 4]:
            if image.shape[0] > scale and image.shape[1] > scale:
                sampled = image[::scale, ::scale]
                
                # Local variance (texture measure)
                variance = np.var(sampled)
                features.append(variance / 10000.0)
                
                # Gradient magnitude
                grad_x = cv2.Sobel(sampled, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(sampled, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                features.append(gradient_magnitude / 1000.0)
                
                # Local binary pattern approximation
                lbp_like = self._compute_lbp_like(sampled)
                features.append(lbp_like)
        
        return features
    
    def _compute_lbp_like(self, image: np.ndarray) -> float:
        """Compute LBP-like texture feature"""
        if image.shape[0] < 3 or image.shape[1] < 3:
            return 0.0
        
        # Simple LBP-like calculation
        center = image[1:-1, 1:-1]
        
        # Compare with 8 neighbors
        neighbors = [
            image[:-2, :-2], image[:-2, 1:-1], image[:-2, 2:],
            image[1:-1, :-2],                  image[1:-1, 2:],
            image[2:, :-2],   image[2:, 1:-1], image[2:, 2:]
        ]
        
        lbp_sum = 0
        for neighbor in neighbors:
            lbp_sum += np.sum(neighbor > center)
        
        return lbp_sum / (center.size * 8)
    
    def _extract_frequency_features(self, image: np.ndarray) -> List[float]:
        """Extract frequency domain features"""
        
        # FFT analysis
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        
        # Frequency statistics
        features = [
            np.mean(fft_magnitude) / 10000.0,
            np.std(fft_magnitude) / 10000.0,
            np.max(fft_magnitude) / 100000.0
        ]
        
        return features
    
    def _count_circles(self, gray_image: np.ndarray) -> int:
        """Count circular shapes (useful for pie charts)"""
        circles = cv2.HoughCircles(
            gray_image, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        return len(circles[0]) if circles is not None else 0
    
    def _is_rectangular(self, contour) -> bool:
        """Check if contour is approximately rectangular"""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return len(approx) == 4 and cv2.contourArea(contour) > 100
    
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
        hex_hash = hex(int(hash_string, 2))[2:].zfill(16)
        
        return hex_hash
    
    def _classify_visual_content(self, image: np.ndarray) -> Tuple[str, Optional[str], Optional[str]]:
        """Classify visual content type using image analysis"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Line detection (charts often have many lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        # Rectangle detection (tables have rectangular structure)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_count = sum(1 for c in contours if self._is_rectangular(c))
        
        # Circle detection (pie charts)
        circle_count = self._count_circles(gray)
        
        # Text density estimation
        text_density = self._estimate_text_density(gray)
        
        # Classification logic
        if rect_count > 6 and line_count > 10 and text_density > 0.2:
            # High rectangle count + lines + text = table
            return VisualContentType.TABLE, None, TableType.DATA_TABLE
        elif rect_count > 3 and line_count > 8 and text_density > 0.15:
            # Medium rectangle count + lines + some text = likely table
            return VisualContentType.TABLE, None, TableType.DATA_TABLE
        elif circle_count > 0 and line_count < 10:
            # Circles with few lines = pie chart
            return VisualContentType.CHART, ChartType.PIE_CHART, None
        elif line_count > 20 and rect_count < 5:
            # Many lines, few rectangles = line chart
            return VisualContentType.CHART, ChartType.LINE_CHART, None
        elif rect_count > 3 and line_count > 5 and text_density < 0.15:
            # Some rectangles + lines + low text = bar chart
            return VisualContentType.CHART, ChartType.BAR_CHART, None
        elif text_density > 0.5:
            # High text density = infographic or diagram
            return VisualContentType.INFOGRAPHIC, None, None
        else:
            # Default to image
            return VisualContentType.IMAGE, None, None
    
    def _estimate_text_density(self, gray_image: np.ndarray) -> float:
        """Estimate text density in image"""
        
        # Use edge density as proxy for text
        edges = cv2.Canny(gray_image, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Text typically has high edge density in small regions
        # Apply morphological operations to detect text-like patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        text_density = np.sum(dilated > 0) / dilated.size
        return min(text_density * 2, 1.0)  # Scale and cap at 1.0
    
    def _extract_data_from_visual(
        self,
        image: np.ndarray,
        content_type: str,
        ocr_text: Optional[str] = None
    ) -> ExtractedData:
        """Extract structured data from visual content"""
        
        # Enhanced logic: Try table extraction first if OCR text suggests tabular data
        if ocr_text and self._has_tabular_structure(ocr_text):
            # Force table extraction if text has clear tabular structure
            return self._extract_table_data(image, ocr_text)
        elif content_type == VisualContentType.TABLE:
            return self._extract_table_data(image, ocr_text)
        elif content_type == VisualContentType.CHART:
            return self._extract_chart_data(image, ocr_text)
        else:
            return self._extract_general_data(image, ocr_text)
    
    def _has_tabular_structure(self, text: str) -> bool:
        """Check if text has clear tabular structure"""
        if not text:
            return False
        
        lines = text.split('\n')
        
        # Look for common table indicators
        table_indicators = [
            '\t',  # Tab-separated values
            'Quarter\t',  # Common financial table headers
            'Revenue\t',
            'Profit\t',
            'Growth\t',
        ]
        
        # Check for tab-separated content
        tab_lines = [line for line in lines if '\t' in line and len(line.split('\t')) > 2]
        if len(tab_lines) >= 2:  # At least header + 1 data row
            return True
        
        # Check for consistent multi-column structure
        multi_col_lines = []
        for line in lines:
            # Split by multiple spaces (common in formatted tables)
            parts = [part.strip() for part in line.split() if part.strip()]
            if len(parts) >= 3:  # At least 3 columns
                multi_col_lines.append(parts)
        
        if len(multi_col_lines) >= 2:  # At least header + 1 data row
            # Check if columns are consistent
            col_counts = [len(parts) for parts in multi_col_lines]
            if len(set(col_counts)) <= 2:  # Allow some variation
                return True
        
        return False
    
    def _extract_table_data(self, image: np.ndarray, ocr_text: Optional[str]) -> ExtractedData:
        """Extract data from table images"""
        
        raw_data = []
        structured_data = {}
        data_quality_score = 0.5
        
        if ocr_text:
            # Parse OCR text for table structure
            lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
            table_rows = []
            
            # Skip title lines (first line if it doesn't contain tabs/multiple columns)
            start_idx = 0
            if lines and len(re.split(r'\s{2,}|\t', lines[0])) == 1:
                start_idx = 1
            
            for line in lines[start_idx:]:
                # Split by tabs first, then by multiple spaces
                if '\t' in line:
                    cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                else:
                    cells = re.split(r'\s{2,}', line.strip())
                    cells = [cell.strip() for cell in cells if cell.strip()]
                
                if len(cells) > 1:  # Valid row with multiple columns
                    table_rows.append(cells)
            
            if table_rows:
                raw_data = table_rows
                
                # Try to identify headers and data
                if len(table_rows) > 1:
                    headers = table_rows[0]
                    data_rows = table_rows[1:]
                    
                    structured_data = {
                        'headers': headers,
                        'rows': data_rows,
                        'num_columns': len(headers),
                        'num_rows': len(data_rows)
                    }
                    
                    data_quality_score = min(0.8, len(table_rows) / 10.0)
                elif len(table_rows) == 1:
                    # Single row, treat as headers
                    headers = table_rows[0]
                    structured_data = {
                        'headers': headers,
                        'rows': [],
                        'num_columns': len(headers),
                        'num_rows': 0
                    }
                    data_quality_score = 0.3
        
        return ExtractedData(
            raw_data=raw_data,
            structured_data=structured_data,
            data_quality_score=data_quality_score
        )
    
    def _extract_chart_data(self, image: np.ndarray, ocr_text: Optional[str]) -> ExtractedData:
        """Extract data from chart images"""
        
        data_points = []
        structured_data = {}
        data_quality_score = 0.3
        
        if ocr_text:
            # Extract numerical values from OCR text
            numbers = re.findall(r'-?\d+\.?\d*', ocr_text)
            
            if numbers:
                # Convert to float values
                numeric_values = []
                for num in numbers:
                    try:
                        numeric_values.append(float(num))
                    except ValueError:
                        continue
                
                if numeric_values:
                    data_points = [{'value': val, 'index': i} for i, val in enumerate(numeric_values)]
                    
                    structured_data = {
                        'values': numeric_values,
                        'min_value': min(numeric_values),
                        'max_value': max(numeric_values),
                        'mean_value': np.mean(numeric_values),
                        'data_points_count': len(numeric_values)
                    }
                    
                    data_quality_score = min(0.7, len(numeric_values) / 20.0)
        
        return ExtractedData(
            data_points=data_points,
            structured_data=structured_data,
            data_quality_score=data_quality_score
        )
    
    def _extract_general_data(self, image: np.ndarray, ocr_text: Optional[str]) -> ExtractedData:
        """Extract general data from other visual content"""
        
        structured_data = {}
        data_quality_score = 0.2
        
        if ocr_text:
            # Basic text analysis
            word_count = len(ocr_text.split())
            char_count = len(ocr_text)
            
            structured_data = {
                'text_content': ocr_text,
                'word_count': word_count,
                'character_count': char_count,
                'has_text': word_count > 0
            }
            
            data_quality_score = min(0.5, word_count / 50.0)
        
        return ExtractedData(
            structured_data=structured_data,
            data_quality_score=data_quality_score
        )
    
    def _analyze_visual_content(
        self,
        image: np.ndarray,
        content_type: str,
        ocr_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze visual content for enhanced metadata extraction"""
        
        analysis = {
            'title': None,
            'caption': None,
            'axis_labels': [],
            'legend_items': [],
            'financial_metrics': [],
            'time_periods': [],
            'currencies': [],
            'business_segments': [],
            'geographic_regions': []
        }
        
        if not ocr_text:
            return analysis
        
        text_lower = ocr_text.lower()
        
        # Extract title (usually first line or largest text)
        lines = ocr_text.split('\n')
        if lines:
            # First non-empty line as potential title
            for line in lines:
                if line.strip():
                    analysis['title'] = line.strip()
                    break
        
        # Extract financial metrics
        for metric in self.financial_terms['metrics']:
            if metric in text_lower:
                analysis['financial_metrics'].append(metric)
        
        # Extract time periods
        for period in self.financial_terms['time_periods']:
            if period in text_lower:
                analysis['time_periods'].append(period)
        
        # Extract currencies
        for currency in self.financial_terms['currencies']:
            if currency in text_lower:
                analysis['currencies'].append(currency)
        
        # Extract business segments
        for segment in self.financial_terms['business_segments']:
            if segment in text_lower:
                analysis['business_segments'].append(segment)
        
        # Extract geographic regions
        for region in self.financial_terms['geographic_regions']:
            if region in text_lower:
                analysis['geographic_regions'].append(region)
        
        # Extract axis labels for charts
        if content_type == VisualContentType.CHART:
            # Look for common axis indicators
            axis_indicators = ['x-axis', 'y-axis', 'horizontal', 'vertical']
            for line in lines:
                line_lower = line.lower()
                if any(indicator in line_lower for indicator in axis_indicators):
                    analysis['axis_labels'].append(line.strip())
        
        # Remove duplicates
        for key in ['financial_metrics', 'time_periods', 'currencies', 'business_segments', 'geographic_regions', 'axis_labels']:
            analysis[key] = list(set(analysis[key]))
        
        return analysis
    
    def _assess_visual_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess visual quality metrics"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        
        # Contrast (standard deviation)
        contrast_score = min(np.std(gray) / 128.0, 1.0)
        
        # Brightness (mean intensity)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
        
        # Overall quality score
        quality_score = (sharpness_score * 0.4 + contrast_score * 0.4 + brightness_score * 0.2)
        
        # Extraction confidence based on image properties
        extraction_confidence = min(quality_score + 0.2, 1.0)
        
        return {
            'quality_score': quality_score,
            'extraction_confidence': extraction_confidence,
            'sharpness': sharpness_score,
            'contrast': contrast_score,
            'brightness': brightness_score
        }
    
    def _calculate_area_percentage(self, bounding_box: Dict[str, float]) -> float:
        """Calculate area percentage of visual element"""
        return bounding_box['width'] * bounding_box['height']
    
    def _encode_image(self, image: np.ndarray) -> Tuple[str, str, Dict[str, int]]:
        """Encode image to base64 for storage"""
        
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Compress and encode
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG', optimize=True)
        
        # Base64 encode
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_data, 'PNG', {'width': image.shape[1], 'height': image.shape[0]}
    
    def _extract_document_context(
        self,
        document_name: Optional[str],
        ocr_text: Optional[str],
        page_context: Optional[str]
    ) -> Dict[str, Any]:
        """Extract document context including bank name, quarter, fiscal year"""
        
        context = {
            'document_name': document_name,
            'document_type': None,
            'bank_name': None,
            'reporting_period': None,
            'fiscal_year': None,
            'quarter': None,
            'section_context': None
        }
        
        # Combine all available text for analysis
        all_text = []
        if document_name:
            all_text.append(document_name.lower())
        if ocr_text:
            all_text.append(ocr_text.lower())
        if page_context:
            all_text.append(page_context.lower())
        
        combined_text = ' '.join(all_text)
        
        if not combined_text:
            return context
        
        # Extract bank name
        for bank in self.financial_terms['banks']:
            if bank in combined_text:
                context['bank_name'] = bank.title()
                break
        
        # Extract quarter information
        quarter_patterns = ['q1', 'q2', 'q3', 'q4', 'first quarter', 'second quarter',
                           'third quarter', 'fourth quarter']
        for pattern in quarter_patterns:
            if pattern in combined_text:
                if pattern.startswith('q'):
                    context['quarter'] = pattern.upper()
                elif 'first' in pattern:
                    context['quarter'] = 'Q1'
                elif 'second' in pattern:
                    context['quarter'] = 'Q2'
                elif 'third' in pattern:
                    context['quarter'] = 'Q3'
                elif 'fourth' in pattern:
                    context['quarter'] = 'Q4'
                break
        
        # Extract fiscal year
        import re
        year_matches = re.findall(r'\b(20\d{2})\b', combined_text)
        if year_matches:
            context['fiscal_year'] = year_matches[-1]  # Take the last/most recent year
        
        # Extract reporting period
        period_patterns = [
            r'(q[1-4]\s+20\d{2})', r'(fy\s*20\d{2})', r'(fiscal\s+year\s+20\d{2})',
            r'(annual\s+report\s+20\d{2})', r'(quarterly\s+report\s+20\d{2})'
        ]
        for pattern in period_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                context['reporting_period'] = matches[0].title()
                break
        
        # Determine document type
        if 'annual' in combined_text:
            context['document_type'] = 'Annual Report'
        elif 'quarterly' in combined_text or context['quarter']:
            context['document_type'] = 'Quarterly Report'
        elif 'earnings' in combined_text:
            context['document_type'] = 'Earnings Report'
        elif '10-k' in combined_text:
            context['document_type'] = '10-K Filing'
        elif '10-q' in combined_text:
            context['document_type'] = '10-Q Filing'
        
        # Extract section context
        section_keywords = [
            'financial highlights', 'income statement', 'balance sheet',
            'cash flow', 'segment performance', 'risk factors', 'outlook',
            'executive summary', 'management discussion', 'md&a'
        ]
        for keyword in section_keywords:
            if keyword in combined_text:
                context['section_context'] = keyword.title()
                break
        
        return context
    
    def _extract_enhanced_visual_metadata(
        self,
        content_type: str,
        ocr_text: Optional[str]
    ) -> Dict[str, Any]:
        """Extract enhanced visual metadata including titles, legends, axis labels"""
        
        metadata = {
            'title': None,
            'subtitle': None,
            'caption': None,
            'x_axis_title': None,
            'y_axis_title': None,
            'axis_labels': [],
            'legend_items': [],
            'legend_title': None,
            'table_headers': [],
            'table_title': None,
            'row_labels': []
        }
        
        if not ocr_text:
            return metadata
        
        lines = ocr_text.split('\n')
        text_lower = ocr_text.lower()
        
        # Extract main title (usually first significant line)
        for line in lines:
            line = line.strip()
            if len(line) > 5 and not line.isdigit():  # Skip short lines and pure numbers
                metadata['title'] = line
                break
        
        # Extract subtitle (second significant line)
        title_found = False
        for line in lines:
            line = line.strip()
            if len(line) > 5 and not line.isdigit():
                if title_found:
                    metadata['subtitle'] = line
                    break
                else:
                    title_found = True
        
        if content_type == 'chart':
            # Extract axis titles
            axis_indicators = [
                ('x-axis', 'x_axis_title'), ('y-axis', 'y_axis_title'),
                ('horizontal', 'x_axis_title'), ('vertical', 'y_axis_title')
            ]
            
            for line in lines:
                line_lower = line.lower().strip()
                for indicator, field in axis_indicators:
                    if indicator in line_lower and len(line.strip()) > len(indicator):
                        # Extract the part after the indicator
                        parts = line.split(':')
                        if len(parts) > 1:
                            metadata[field] = parts[1].strip()
                        else:
                            metadata[field] = line.strip()
            
            # Extract legend items
            legend_keywords = ['legend', 'key', 'series']
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in legend_keywords):
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            metadata['legend_title'] = parts[0].strip()
                            metadata['legend_items'].append(parts[1].strip())
                    else:
                        metadata['legend_items'].append(line.strip())
            
            # Extract data series from chart
            # Look for patterns like "Series 1", "Product A", etc.
            series_patterns = [
                r'([A-Za-z\s]+):\s*[\d\.,]+[%$]?',  # "Product A: 25%"
                r'([A-Za-z\s]+)\s+[\d\.,]+[%$]?',   # "Product A 25%"
            ]
            
            for pattern in series_patterns:
                matches = re.findall(pattern, ocr_text)
                for match in matches:
                    if len(match.strip()) > 2:
                        metadata['legend_items'].append(match.strip())
        
        elif content_type == 'table':
            # Extract table headers (first row or lines with consistent formatting)
            potential_headers = []
            
            for line in lines:
                line = line.strip()
                if line and not line.replace(' ', '').replace('\t', '').isdigit():
                    # Split by multiple spaces or tabs
                    cells = re.split(r'\s{2,}|\t', line)
                    if len(cells) > 1:
                        potential_headers = cells
                        break
            
            if potential_headers:
                metadata['table_headers'] = [h.strip() for h in potential_headers if h.strip()]
            
            # Extract table title (look for lines before the data)
            for i, line in enumerate(lines):
                line = line.strip()
                if line and len(line) > 10 and not any(char.isdigit() for char in line[:5]):
                    # Check if next few lines contain tabular data
                    has_tabular_data = False
                    for j in range(i+1, min(i+4, len(lines))):
                        if '\t' in lines[j] or re.search(r'\s{2,}', lines[j]):
                            has_tabular_data = True
                            break
                    
                    if has_tabular_data:
                        metadata['table_title'] = line
                        break
            
            # Extract row labels (first column of data rows)
            for line in lines:
                line = line.strip()
                if line and '\t' in line or re.search(r'\s{2,}', line):
                    cells = re.split(r'\s{2,}|\t', line)
                    if len(cells) > 1 and cells[0].strip():
                        first_cell = cells[0].strip()
                        # Check if it's likely a row label (not just numbers)
                        if not first_cell.replace('.', '').replace(',', '').isdigit():
                            metadata['row_labels'].append(first_cell)
        
        # Clean up and deduplicate
        for key in ['axis_labels', 'legend_items', 'table_headers', 'row_labels']:
            if metadata[key]:
                metadata[key] = list(dict.fromkeys(metadata[key]))  # Remove duplicates while preserving order
        
        return metadata