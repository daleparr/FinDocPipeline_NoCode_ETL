# FinDocPipeline - Deployment Guide

## 🚀 Quick Start Deployment

### **Option 1: Local Development**
```bash
# Clone the repository
git clone https://github.com/daleparr/FinDocPipeline_NoCode_ETL.git
cd FinDocPipeline_NoCode_ETL

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run FinDocPipeline.py
```

### **Option 2: Streamlit Cloud (Recommended)**
1. **Fork the repository** to your GitHub account
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Click** "New app"
4. **Select** your forked repository
5. **Set main file** to `FinDocPipeline.py`
6. **Deploy** - your app will be live in minutes!

### **Option 3: Docker Deployment**
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "FinDocPipeline.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t findocpipeline .
docker run -p 8501:8501 findocpipeline
```

## 📋 System Requirements

### **Minimum Requirements**
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM
- **Storage**: 1GB free space
- **OS**: Windows, macOS, or Linux

### **Recommended Setup**
- **Python**: 3.9+
- **Memory**: 8GB+ RAM for large documents
- **CPU**: Multi-core processor for faster processing
- **Storage**: SSD for improved I/O performance

## 🔧 Dependencies Installation

### **Core Dependencies**
```bash
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install pdfplumber>=0.9.0
pip install PyMuPDF>=1.23.0
```

### **Computer Vision & OCR**
```bash
pip install opencv-python>=4.8.0
pip install pytesseract>=0.3.10
pip install Pillow>=10.0.0
pip install numpy>=1.24.0
```

### **System Dependencies (Linux/Ubuntu)**
```bash
# For OCR functionality
sudo apt-get install tesseract-ocr

# For OpenCV
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### **System Dependencies (macOS)**
```bash
# Using Homebrew
brew install tesseract
```

### **System Dependencies (Windows)**
- **Tesseract**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **Add to PATH**: Ensure tesseract.exe is in your system PATH

## 🌐 Production Deployment

### **Streamlit Cloud Configuration**
Create `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### **Environment Variables**
```bash
# Optional: Set custom configurations
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### **Performance Optimization**
```python
# Add to FinDocPipeline.py for production
import streamlit as st

# Cache configuration
@st.cache_data
def load_patterns():
    # Cache regex patterns for better performance
    pass

# Memory management
st.set_page_config(
    page_title="FinDocPipeline",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

## 🔒 Security Considerations

### **File Upload Security**
- **File Size Limit**: 200MB maximum (configurable)
- **File Type Validation**: Only PDF files accepted
- **Temporary Files**: Automatically cleaned after processing
- **No Persistent Storage**: Files not saved on server

### **Data Privacy**
- **No Data Retention**: Uploaded files processed and deleted
- **Local Processing**: All processing happens on your server
- **No External APIs**: No data sent to third-party services

## 🚀 Streamlit Cloud Deployment Steps

### **Step 1: Prepare Repository**
✅ Repository is ready at: `https://github.com/daleparr/FinDocPipeline_NoCode_ETL.git`

### **Step 2: Deploy to Streamlit Cloud**
1. **Visit**: [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with GitHub
3. **Click** "New app"
4. **Repository**: `daleparr/FinDocPipeline_NoCode_ETL`
5. **Branch**: `master`
6. **Main file path**: `FinDocPipeline.py`
7. **Click** "Deploy!"

### **Step 3: Configure (Optional)**
- **Custom domain**: Available with Streamlit Cloud Pro
- **Environment variables**: Set in app settings
- **Resource limits**: Automatic scaling

## 🔧 Troubleshooting

### **Common Issues**

#### **"ModuleNotFoundError"**
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### **"Tesseract not found"**
```bash
# Linux/Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download and install from GitHub, add to PATH
```

#### **"Memory Error"**
- **Reduce file size**: Keep PDFs under 100MB
- **Increase system memory**: 8GB+ recommended
- **Process in chunks**: For very large documents

#### **"Streamlit Cloud Build Failed"**
- **Check requirements.txt**: Ensure all dependencies listed
- **Python version**: Streamlit Cloud uses Python 3.9
- **File paths**: Use relative paths only

### **Performance Issues**
- **Slow processing**: Normal for large documents (2-5 minutes)
- **Memory usage**: Expected for PDF processing
- **OCR timeout**: Disable OCR for faster processing if not needed

## 📊 Monitoring & Analytics

### **Built-in Metrics**
- **Processing time**: Displayed in UI
- **Success rate**: Error handling and reporting
- **File statistics**: Pages, words, tables, metrics extracted

### **Custom Analytics (Optional)**
```python
# Add to FinDocPipeline.py
import time
import logging

# Performance logging
start_time = time.time()
# ... processing ...
processing_time = time.time() - start_time
st.info(f"Processing completed in {processing_time:.2f} seconds")
```

## 🎯 Production Checklist

- [ ] **Dependencies installed**: All requirements.txt packages
- [ ] **System dependencies**: Tesseract OCR installed
- [ ] **File upload working**: Test with sample PDF
- [ ] **OCR functionality**: Computer vision capabilities available
- [ ] **Export functions**: All download buttons working
- [ ] **Error handling**: Graceful failure for invalid files
- [ ] **Performance**: Acceptable processing times
- [ ] **Security**: File cleanup and validation working
- [ ] **Documentation**: All guides accessible
- [ ] **Customization**: UI styling as desired

## 🌟 Post-Deployment

### **Share Your App**
- **Public URL**: Available immediately after Streamlit Cloud deployment
- **Custom domain**: Configure in Streamlit Cloud settings
- **Social sharing**: Share the URL with stakeholders

### **Maintenance**
- **Updates**: Push to GitHub, auto-deploys to Streamlit Cloud
- **Monitoring**: Check app health via Streamlit Cloud dashboard
- **Scaling**: Automatic with Streamlit Cloud

---

**Your app is now live at**: `https://your-app-name.streamlit.app`

**Repository**: `https://github.com/daleparr/FinDocPipeline_NoCode_ETL.git`