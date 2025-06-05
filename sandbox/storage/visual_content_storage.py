"""
Visual Content Storage for FinDocPipeline
Streamlit-optimized storage for visual content embeddings and metadata.
"""

import streamlit as st
import json
import pickle
import base64
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from schemas.visual_content_schema import VisualContentSchema

class VisualContentStorage:
    """Streamlit-optimized visual content storage with embedding search"""
    
    def __init__(self):
        self.storage_key = "visual_content_storage"
        self.embedding_key = "visual_embeddings"
        self.similarity_index_key = "visual_similarity_index"
        self.metadata_key = "visual_metadata"
        
        # Initialize storage in session state
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage containers in session state"""
        
        if self.storage_key not in st.session_state:
            st.session_state[self.storage_key] = {}
        
        if self.embedding_key not in st.session_state:
            st.session_state[self.embedding_key] = {}
        
        if self.similarity_index_key not in st.session_state:
            st.session_state[self.similarity_index_key] = {}
        
        if self.metadata_key not in st.session_state:
            st.session_state[self.metadata_key] = {
                'total_visuals': 0,
                'last_updated': None,
                'content_type_counts': {},
                'document_counts': {}
            }
    
    def store_visual_content(self, visual_content: VisualContentSchema) -> None:
        """Store visual content with embeddings and update indices"""
        
        # Store main content
        content_dict = visual_content.dict()
        st.session_state[self.storage_key][visual_content.visual_id] = content_dict
        
        # Store embedding separately for efficient similarity search
        st.session_state[self.embedding_key][visual_content.visual_id] = {
            'embedding_vector': visual_content.visual_embedding.embedding_vector,
            'similarity_hash': visual_content.visual_embedding.similarity_hash,
            'content_type': visual_content.content_type,
            'document_id': visual_content.document_id,
            'page_number': visual_content.page_number,
            'quality_score': visual_content.image_quality_score
        }
        
        # Update similarity index
        self._update_similarity_index(visual_content)
        
        # Update metadata
        self._update_metadata(visual_content)
    
    def get_visual_content(self, visual_id: str) -> Optional[VisualContentSchema]:
        """Retrieve visual content by ID"""
        
        content_data = st.session_state[self.storage_key].get(visual_id)
        if content_data:
            return VisualContentSchema(**content_data)
        return None
    
    def get_all_visual_content(self) -> List[VisualContentSchema]:
        """Get all stored visual content"""
        
        visuals = []
        for content_data in st.session_state[self.storage_key].values():
            visuals.append(VisualContentSchema(**content_data))
        
        return visuals
    
    def find_similar_visuals(
        self, 
        visual_id: str, 
        similarity_threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Find visually similar content using embedding similarity"""
        
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
                    'document_id': other_embedding['document_id'],
                    'page_number': other_embedding['page_number'],
                    'quality_score': other_embedding['quality_score']
                })
        
        # Sort by similarity and return top results
        similar_visuals.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_visuals[:max_results]
    
    def find_similar_by_hash(
        self, 
        visual_id: str, 
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar visuals using perceptual hash"""
        
        target_embedding = st.session_state[self.embedding_key].get(visual_id)
        if not target_embedding:
            return []
        
        target_hash = target_embedding['similarity_hash']
        similar_visuals = []
        
        for other_id, other_embedding in st.session_state[self.embedding_key].items():
            if other_id == visual_id:
                continue
            
            # Calculate Hamming distance between hashes
            other_hash = other_embedding['similarity_hash']
            hamming_distance = self._hamming_distance(target_hash, other_hash)
            
            # Convert to similarity score (lower distance = higher similarity)
            similarity_score = 1.0 - (hamming_distance / len(target_hash))
            
            if similarity_score > 0.7:  # Threshold for hash similarity
                similar_visuals.append({
                    'visual_id': other_id,
                    'similarity_score': similarity_score,
                    'content_type': other_embedding['content_type'],
                    'document_id': other_embedding['document_id'],
                    'hamming_distance': hamming_distance
                })
        
        # Sort by similarity
        similar_visuals.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_visuals[:max_results]
    
    def get_visuals_by_document(self, document_id: str) -> List[VisualContentSchema]:
        """Get all visual content for a specific document"""
        
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
    
    def search_visuals_by_metadata(
        self, 
        search_terms: List[str],
        search_fields: List[str] = None
    ) -> List[VisualContentSchema]:
        """Search visuals by metadata content"""
        
        if search_fields is None:
            search_fields = ['title', 'caption', 'financial_metrics', 'time_periods']
        
        matching_visuals = []
        search_terms_lower = [term.lower() for term in search_terms]
        
        for visual_data in st.session_state[self.storage_key].values():
            match_found = False
            
            for field in search_fields:
                if field in visual_data and visual_data[field]:
                    field_content = str(visual_data[field]).lower()
                    
                    if any(term in field_content for term in search_terms_lower):
                        match_found = True
                        break
            
            if match_found:
                matching_visuals.append(VisualContentSchema(**visual_data))
        
        return matching_visuals
    
    def get_visual_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored visual content"""
        
        metadata = st.session_state[self.metadata_key]
        
        # Calculate additional statistics
        all_visuals = list(st.session_state[self.storage_key].values())
        
        if not all_visuals:
            return {
                'total_visuals': 0,
                'content_types': {},
                'documents': {},
                'quality_stats': {},
                'processing_stats': {}
            }
        
        # Quality statistics
        quality_scores = [v['image_quality_score'] for v in all_visuals]
        confidence_scores = [v['extraction_confidence'] for v in all_visuals]
        processing_times = [v['processing_time'] for v in all_visuals]
        
        quality_stats = {
            'avg_quality': np.mean(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'avg_confidence': np.mean(confidence_scores),
            'avg_processing_time': np.mean(processing_times)
        }
        
        # Content type distribution
        content_type_counts = {}
        for visual in all_visuals:
            content_type = visual['content_type']
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        # Document distribution
        document_counts = {}
        for visual in all_visuals:
            doc_id = visual['document_id']
            document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
        
        return {
            'total_visuals': len(all_visuals),
            'content_types': content_type_counts,
            'documents': document_counts,
            'quality_stats': quality_stats,
            'processing_stats': {
                'total_processing_time': sum(processing_times),
                'avg_processing_time': np.mean(processing_times)
            },
            'last_updated': metadata['last_updated']
        }
    
    def export_visual_metadata(self) -> Dict[str, Any]:
        """Export all visual metadata for analysis"""
        
        export_data = {
            'visual_content': st.session_state[self.storage_key],
            'embeddings': st.session_state[self.embedding_key],
            'similarity_index': st.session_state[self.similarity_index_key],
            'metadata': st.session_state[self.metadata_key],
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.get_visual_statistics()
        }
        
        return export_data
    
    def clear_storage(self) -> None:
        """Clear all visual content storage"""
        
        st.session_state[self.storage_key] = {}
        st.session_state[self.embedding_key] = {}
        st.session_state[self.similarity_index_key] = {}
        st.session_state[self.metadata_key] = {
            'total_visuals': 0,
            'last_updated': None,
            'content_type_counts': {},
            'document_counts': {}
        }
    
    def _update_similarity_index(self, visual_content: VisualContentSchema) -> None:
        """Update similarity index for fast lookups"""
        
        # Group by content type for faster similarity searches
        content_type = visual_content.content_type
        
        if content_type not in st.session_state[self.similarity_index_key]:
            st.session_state[self.similarity_index_key][content_type] = []
        
        st.session_state[self.similarity_index_key][content_type].append({
            'visual_id': visual_content.visual_id,
            'document_id': visual_content.document_id,
            'quality_score': visual_content.image_quality_score
        })
    
    def _update_metadata(self, visual_content: VisualContentSchema) -> None:
        """Update storage metadata"""
        
        metadata = st.session_state[self.metadata_key]
        
        # Update counts
        metadata['total_visuals'] += 1
        metadata['last_updated'] = datetime.now().isoformat()
        
        # Update content type counts
        content_type = visual_content.content_type
        if content_type not in metadata['content_type_counts']:
            metadata['content_type_counts'][content_type] = 0
        metadata['content_type_counts'][content_type] += 1
        
        # Update document counts
        doc_id = visual_content.document_id
        if doc_id not in metadata['document_counts']:
            metadata['document_counts'][doc_id] = 0
        metadata['document_counts'][doc_id] += 1
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hash strings"""
        
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def get_embedding_clusters(self, n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster visual embeddings for analysis"""
        
        if len(st.session_state[self.embedding_key]) < n_clusters:
            return {'error': 'Not enough visuals for clustering'}
        
        try:
            from sklearn.cluster import KMeans
            
            # Collect all embeddings
            embeddings = []
            visual_ids = []
            
            for visual_id, embedding_data in st.session_state[self.embedding_key].items():
                embeddings.append(embedding_data['embedding_vector'])
                visual_ids.append(visual_id)
            
            embeddings = np.array(embeddings)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Organize results
            clusters = {}
            for i, (visual_id, label) in enumerate(zip(visual_ids, cluster_labels)):
                if label not in clusters:
                    clusters[label] = []
                
                embedding_data = st.session_state[self.embedding_key][visual_id]
                clusters[label].append({
                    'visual_id': visual_id,
                    'content_type': embedding_data['content_type'],
                    'document_id': embedding_data['document_id'],
                    'quality_score': embedding_data['quality_score']
                })
            
            return {
                'clusters': clusters,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'n_clusters': n_clusters,
                'total_visuals': len(visual_ids)
            }
            
        except ImportError:
            return {'error': 'Clustering requires scikit-learn'}
        except Exception as e:
            return {'error': f'Clustering failed: {str(e)}'}