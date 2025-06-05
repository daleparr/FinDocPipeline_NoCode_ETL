# 🧪 FinDocPipeline Enhanced - Sandbox Environment

## 📋 Overview

This sandbox contains the enhanced multi-document version of FinDocPipeline that extends the existing sophisticated single-document system with concurrent processing capabilities.

## 🚀 Key Features

### **Multi-Document Processing**
- Process up to 20 documents concurrently
- 4 worker threads optimized for Streamlit
- Real-time progress tracking and status updates
- Batch processing with intelligent resource management

### **Enhanced File Support**
- **PDF**: Uses existing sophisticated ComprehensiveFinancialParser
- **DOCX**: New Microsoft Word document processing
- **Excel**: XLSX and XLS file support with financial pattern detection
- **CSV**: Structured data processing

### **AI-Powered Classification**
- Automatic document type detection (transcript, presentation, financial summary)
- Rule-based classification with high accuracy
- Confidence scoring and reasoning

### **Advanced Analytics**
- Leverages existing NLP processing pipeline
- Financial metrics extraction with deduplication
- Visual content analysis (charts, graphs, images)
- Cross-document comparative analysis

## 🏗️ Architecture

### **Extends Existing Components**
The sandbox **reuses** all existing sophisticated components:
- `ComprehensiveFinancialParser` - Advanced PDF processing
- `NLPDataProcessor` - Text cleaning and NLP preparation  
- `NLPDatasetExporter` - Feature extraction and labeling
- `EnhancedVisualParser` - Computer vision and OCR
- `DeduplicatedMetricsExtractor` - Financial metrics extraction

### **Adds New Multi-Document Components**
- `MultiDocumentProcessor` - Concurrent processing orchestrator
- `DocumentClassifier` - Lightweight document type classification
- `ConcurrentProcessingManager` - Resource-aware concurrent execution
- `BaseDocumentSchema` - Enhanced data structures

## 🚀 Running the Sandbox

### **Prerequisites**
```bash
# Install enhanced dependencies
pip install -r requirements_enhanced.txt
```

### **Launch Enhanced Version**
```bash
# From the sandbox directory
cd sandbox
streamlit run FinDocPipeline_Enhanced.py --server.port 8502
```

### **Keep Live Version Running**
```bash
# From the main directory (unchanged)
streamlit run FinDocPipeline.py
```

## 📊 Usage Guide

### **1. Upload Multiple Documents**
- Drag and drop up to 20 files (PDF, DOCX, Excel, CSV)
- View file queue with automatic type detection
- Set processing priorities if needed

### **2. Process Documents**
- Click "🚀 Process All Documents" 
- Watch real-time progress with live updates
- View processing statistics and success rates

### **3. Analyze Results**
- Comprehensive summary with content metrics
- Document type distribution analysis
- Individual document processing details
- Export results in JSON or CSV format

### **4. Advanced Features**
- Cross-document analytics and comparisons
- Financial metrics aggregation across documents
- Visual content analysis (charts, tables, images)
- NLP dataset generation for machine learning

## 🔧 Technical Details

### **Concurrent Processing**
- **Max Workers**: 4 (Streamlit optimized)
- **Batch Size**: 5 documents per batch
- **Timeout**: 2 minutes per document
- **Resource Monitoring**: Memory and CPU usage tracking

### **File Size Limits**
- **PDF**: 10MB maximum
- **DOCX**: 10MB maximum  
- **Excel**: 10MB maximum
- **CSV**: 5MB maximum

### **Performance Targets**
- **Processing Speed**: <30 seconds per document
- **Memory Usage**: <1GB total system memory
- **Success Rate**: >95% for supported file types
- **Classification Accuracy**: >90% for financial documents

## 📈 Comparison with Live Version

| Feature | Live Version | Enhanced Sandbox |
|---------|-------------|------------------|
| **Documents** | 1 at a time | Up to 20 concurrent |
| **File Types** | PDF only | PDF, DOCX, Excel, CSV |
| **Processing** | Sequential | Concurrent (4 workers) |
| **Classification** | Manual | Automatic AI-powered |
| **Analytics** | Single document | Cross-document analysis |
| **Export** | Basic CSV/JSON | Enhanced multi-format |
| **Progress** | Simple | Real-time dashboard |

## 🛡️ Safety Features

### **Sandbox Isolation**
- Completely separate from live system
- No impact on existing functionality
- Independent port (8502 vs 8501)
- Separate requirements and dependencies

### **Error Handling**
- Graceful degradation on processing failures
- Comprehensive error logging and reporting
- Automatic retry mechanisms
- Resource cleanup and memory management

### **Data Validation**
- File type and size validation
- Content integrity checking
- Schema validation for processed data
- Quality assurance metrics

## 🔄 Migration Path

When ready to migrate features to the live system:

1. **Selective Feature Migration**: Choose specific enhancements to migrate
2. **Gradual Rollout**: Test individual components before full integration
3. **Backward Compatibility**: Maintain existing single-document functionality
4. **User Choice**: Allow users to choose between single and multi-document modes

## 📋 Development Status

### **✅ Completed**
- Multi-document processing architecture
- File handler system (PDF, DOCX, Excel, CSV)
- Document classification engine
- Concurrent processing manager
- Enhanced UI with progress tracking
- Integration with existing sophisticated processors

### **🔄 In Progress**
- Visual content processing enhancements
- Advanced schema implementations
- Cross-document analytics dashboard
- Performance optimization

### **📋 Planned**
- Advanced ML-based classification
- Custom document type training
- API endpoints for external integration
- Advanced visualization components

## 🆘 Troubleshooting

### **Common Issues**
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or file count
3. **Processing Timeouts**: Check file sizes and complexity
4. **Classification Errors**: Verify file formats are supported

### **Performance Tips**
1. **Optimal Batch Size**: 5-10 documents for best performance
2. **File Preparation**: Ensure files are not corrupted
3. **Resource Monitoring**: Watch memory usage during processing
4. **Network Stability**: Ensure stable connection for large uploads

## 📞 Support

For issues or questions about the enhanced sandbox:
1. Check existing documentation in the main project
2. Review error messages and logs
3. Test with smaller file sets first
4. Compare behavior with live version

The sandbox is designed to be a safe testing environment for the enhanced multi-document capabilities while preserving the stability and functionality of the existing live system.