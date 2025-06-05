# 🧪 FinDocPipeline Enhanced - Sandbox Environment

## 📋 **Sandbox Overview**

This sandbox environment contains the enhanced FinDocPipeline with multi-document concurrent processing, rich schemas, and visual content capabilities. The sandbox is completely isolated from the live solution to prevent any disruption.

## 📁 **Sandbox Structure**

```
FinDocPipeline/                     # Main project (LIVE - untouched)
├── FinDocPipeline.py              # Current live application
├── requirements.txt               # Current dependencies
└── ... (all existing files)       # Live solution files

sandbox/                           # NEW: Enhanced version sandbox
├── FinDocPipeline_Enhanced.py     # Enhanced multi-document application
├── requirements_enhanced.txt      # Enhanced dependencies
├── README_SANDBOX.md              # Sandbox documentation
├── 
├── file_handlers/                 # Multi-format file processing
│   ├── __init__.py
│   ├── base_handler.py
│   ├── pdf_handler.py
│   ├── docx_handler.py
│   ├── excel_handler.py
│   └── csv_handler.py
│
├── classification/                # Document classification
│   ├── __init__.py
│   ├── document_classifier.py
│   └── feature_extractors.py
│
├── schemas/                       # Enhanced schemas
│   ├── __init__.py
│   ├── base_schema.py
│   ├── transcript_schema.py
│   ├── presentation_schema.py
│   └── summary_schema.py
│
├── processing/                    # Concurrent processing
│   ├── __init__.py
│   ├── concurrent_manager.py
│   ├── transcript_processor.py
│   ├── presentation_processor.py
│   └── summary_processor.py
│
├── visual/                        # Visual content processing
│   ├── __init__.py
│   ├── visual_processor.py
│   ├── embedding_generator.py
│   └── similarity_engine.py
│
├── ui/                           # Enhanced UI components
│   ├── __init__.py
│   ├── multi_upload.py
│   ├── progress_dashboard.py
│   └── visual_analytics.py
│
├── storage/                      # Data storage
│   ├── __init__.py
│   ├── session_storage.py
│   └── visual_storage.py
│
├── utils/                        # Utilities
│   ├── __init__.py
│   ├── text_processing.py
│   └── export_utils.py
│
└── tests/                        # Test suite
    ├── __init__.py
    ├── test_file_handlers.py
    ├── test_classification.py
    └── sample_data/
```

## 🚀 **Running the Sandbox**

### **Live Solution (Unchanged)**
```bash
# Continue using the live solution as normal
streamlit run FinDocPipeline.py
```

### **Enhanced Sandbox Version**
```bash
# Run the enhanced version from sandbox
cd sandbox
streamlit run FinDocPipeline_Enhanced.py --server.port 8502
```

## 🔧 **Key Features**

### **Multi-Document Processing**
- Process up to 20 documents concurrently
- 4 worker threads for optimal Streamlit performance
- Batch processing with real-time progress tracking

### **Enhanced Schemas**
- Rich metadata extraction for transcripts, presentations, summaries
- Speaker identification and Q&A parsing
- Financial metrics and trend analysis

### **Visual Content Processing**
- Chart and table detection with embeddings
- Similarity search for visual content
- Comprehensive visual analytics dashboard

### **Streamlit Compatibility**
- Lightweight dependencies only
- Session state management
- Cached processing for performance
- No system-level dependencies

## 📋 **Development Workflow**

1. **Develop in Sandbox**: All new features developed in `sandbox/`
2. **Test Thoroughly**: Comprehensive testing before any migration
3. **Keep Live Stable**: Live solution remains completely untouched
4. **Gradual Migration**: When ready, features can be selectively migrated