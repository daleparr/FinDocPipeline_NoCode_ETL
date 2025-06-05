"""
Concurrent processing manager for multi-document processing.
Streamlit-optimized with resource management and progress tracking.
"""

import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import psutil
from typing import List, Dict, Any, Callable, Optional
import queue

class ConcurrentProcessingManager:
    """
    Streamlit-safe concurrent processing manager with resource monitoring.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = min(max_workers, 4)  # Streamlit limit
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Resource monitoring
        self.memory_threshold_mb = 1024  # 1GB limit
        self.cpu_threshold = 90  # 90% CPU limit
        
        # Progress tracking
        self.progress_queue = queue.Queue()
        self.task_counter = 0
        
    def process_batch(
        self, 
        items: List[Any], 
        processor_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of items concurrently with resource monitoring.
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing processing results
        """
        
        if not items:
            return {'results': {}, 'stats': {'total': 0, 'completed': 0, 'failed': 0}}
        
        # Check system resources before starting
        if not self._check_system_resources():
            st.warning("⚠️ System resources are limited. Processing may be slower.")
        
        results = {}
        stats = {'total': len(items), 'completed': 0, 'failed': 0, 'start_time': time.time()}
        
        # Process with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {}
            for item in items:
                task_id = f"task_{self.task_counter}"
                self.task_counter += 1
                
                future = executor.submit(self._safe_process_item, task_id, item, processor_func)
                future_to_item[future] = {'item': item, 'task_id': task_id}
                
                self.active_tasks[task_id] = {
                    'item': item,
                    'start_time': time.time(),
                    'status': 'processing'
                }
            
            # Collect results as they complete
            for future in as_completed(future_to_item, timeout=300):  # 5 minute timeout
                item_info = future_to_item[future]
                item = item_info['item']
                task_id = item_info['task_id']
                
                try:
                    result = future.result()
                    
                    # Determine item identifier
                    item_key = getattr(item, 'name', str(item))
                    results[item_key] = result
                    
                    # Update stats
                    if result.get('status') == 'completed':
                        stats['completed'] += 1
                    else:
                        stats['failed'] += 1
                    
                    # Move to completed tasks
                    self.completed_tasks[task_id] = self.active_tasks[task_id]
                    self.completed_tasks[task_id]['result'] = result
                    self.completed_tasks[task_id]['end_time'] = time.time()
                    del self.active_tasks[task_id]
                    
                    # Call progress callback
                    if progress_callback:
                        progress_callback(item_key, result, None)
                
                except Exception as e:
                    # Handle failed task
                    item_key = getattr(item, 'name', str(item))
                    error_result = {
                        'status': 'failed',
                        'error': str(e),
                        'item': item_key
                    }
                    results[item_key] = error_result
                    stats['failed'] += 1
                    
                    # Move to failed tasks
                    self.failed_tasks[task_id] = self.active_tasks[task_id]
                    self.failed_tasks[task_id]['error'] = str(e)
                    self.failed_tasks[task_id]['end_time'] = time.time()
                    del self.active_tasks[task_id]
                    
                    # Call progress callback for error
                    if progress_callback:
                        progress_callback(item_key, None, str(e))
        
        # Calculate final stats
        stats['end_time'] = time.time()
        stats['total_time'] = stats['end_time'] - stats['start_time']
        stats['success_rate'] = stats['completed'] / stats['total'] if stats['total'] > 0 else 0
        
        return {'results': results, 'stats': stats}
    
    def _safe_process_item(self, task_id: str, item: Any, processor_func: Callable) -> Dict[str, Any]:
        """
        Safely process a single item with error handling and resource monitoring.
        
        Args:
            task_id: Unique task identifier
            item: Item to process
            processor_func: Processing function
            
        Returns:
            Dict containing processing result
        """
        
        start_time = time.time()
        
        try:
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'processing'
            
            # Monitor resources during processing
            if not self._check_system_resources():
                # If resources are low, add a small delay
                time.sleep(0.5)
            
            # Process the item
            result = processor_func(item)
            
            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {'data': result}
            
            result['status'] = result.get('status', 'completed')
            result['processing_time'] = time.time() - start_time
            result['task_id'] = task_id
            
            return result
        
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'task_id': task_id
            }
    
    def _check_system_resources(self) -> bool:
        """
        Check if system has sufficient resources for processing.
        
        Returns:
            bool: True if resources are sufficient
        """
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = (memory.total - memory.available) / (1024 * 1024)
            
            if memory_usage_mb > self.memory_threshold_mb:
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_threshold:
                return False
            
            return True
        
        except Exception:
            # If we can't check resources, assume they're okay
            return True
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_tasks': self.task_counter,
            'max_workers': self.max_workers,
            'system_resources': self._get_system_resource_info()
        }
    
    def _get_system_resource_info(self) -> Dict[str, Any]:
        """Get current system resource information"""
        
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'cpu_count': psutil.cpu_count()
            }
        
        except Exception as e:
            return {'error': f'Could not get system info: {str(e)}'}
    
    def clear_completed_tasks(self):
        """Clear completed and failed task history"""
        self.completed_tasks.clear()
        self.failed_tasks.clear()
    
    def get_active_task_info(self) -> List[Dict[str, Any]]:
        """Get information about currently active tasks"""
        
        active_info = []
        current_time = time.time()
        
        for task_id, task_info in self.active_tasks.items():
            elapsed_time = current_time - task_info['start_time']
            
            active_info.append({
                'task_id': task_id,
                'item': str(task_info['item'])[:50],  # Truncate for display
                'status': task_info['status'],
                'elapsed_time': round(elapsed_time, 2),
                'start_time': task_info['start_time']
            })
        
        return active_info
    
    @st.cache_data
    def estimate_processing_time(_self, num_items: int, avg_item_time: float = 10.0) -> Dict[str, Any]:
        """
        Estimate total processing time for a batch.
        
        Args:
            num_items: Number of items to process
            avg_item_time: Average processing time per item in seconds
            
        Returns:
            Dict containing time estimates
        """
        
        # Calculate concurrent processing time
        batches = (num_items + _self.max_workers - 1) // _self.max_workers
        concurrent_time = batches * avg_item_time
        
        # Calculate sequential processing time for comparison
        sequential_time = num_items * avg_item_time
        
        # Add overhead for task management
        overhead_time = num_items * 0.5  # 0.5 seconds overhead per item
        total_estimated_time = concurrent_time + overhead_time
        
        return {
            'estimated_total_time': round(total_estimated_time, 1),
            'concurrent_time': round(concurrent_time, 1),
            'sequential_time': round(sequential_time, 1),
            'time_saved': round(sequential_time - total_estimated_time, 1),
            'efficiency_gain': round((sequential_time - total_estimated_time) / sequential_time * 100, 1) if sequential_time > 0 else 0,
            'batches_required': batches,
            'items_per_batch': _self.max_workers
        }