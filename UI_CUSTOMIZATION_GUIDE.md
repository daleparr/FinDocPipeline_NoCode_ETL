# FinDocPipeline - UI Customization Guide

## 🎨 CSS Customization Locations

### **Primary CSS Location: FinDocPipeline.py**
The main CSS customization is located in `FinDocPipeline.py` starting around **line 960** in the `st.markdown()` block with `unsafe_allow_html=True`.

```python
# Custom CSS for UI customization
st.markdown("""
<style>
/* Your custom CSS goes here */
</style>
""", unsafe_allow_html=True)
```

## 🔧 Key UI Elements You Can Customize

### **1. Main Title & Branding**
```css
/* Main title styling */
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;  /* Change this color */
    margin-bottom: 0.5rem;
}
```

**To customize**: Change the `color` value to your brand color.

### **2. Subtitle/Tagline**
```css
/* Subtitle styling */
.subtitle {
    font-size: 1.2rem;
    color: #666;  /* Change this color */
    margin-bottom: 2rem;
    font-style: italic;  /* Remove italic if desired */
}
```

### **3. ETL Processing Messages**
```css
/* ETL Pipeline step styling */
.etl-step {
    background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
    padding: 0.5rem 1rem;
    border-left: 4px solid #1f77b4;  /* Change accent color */
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
}
```

### **4. Success Messages**
```css
/* Success message styling */
.success-message {
    background: linear-gradient(90deg, #f0fff0 0%, #e6ffe6 100%);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;  /* Success color */
    margin: 1rem 0;
}
```

### **5. Download Buttons**
```css
/* Custom button styling */
.stDownloadButton > button {
    background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
```

## 📝 Language Customization Points

### **1. Processing Status Messages**
Located around **line 1009-1026** in `FinDocPipeline.py`:

```python
with st.spinner("📊 EXTRACT: Reading slide deck content..."):
with st.spinner("🔄 TRANSFORM: Creating raw structured data..."):
with st.spinner("🧹 TRANSFORM: Cleaning text and normalizing tokens..."):
with st.spinner("🎯 TRANSFORM: Applying METRIC_PATTERNS regex library..."):
with st.spinner("👁️ TRANSFORM: Processing visual content with OCR..."):
with st.spinner("📋 LOAD: Generating clean datasets for analysis..."):
```

### **2. Success Message**
Located around **line 1029**:
```python
st.success(f"✅ ETL Pipeline Complete: {uploaded_file.name} ({len(pages_data)} slides processed)")
```

### **3. Dashboard Headers**
Located around **line 1032**:
```python
st.header("📊 Your Clean, Analysis-Ready Datasets")
```

### **4. Metric Labels**
Located around **lines 1037-1044**:
```python
st.metric("Slides Processed", len(pages_data))
st.metric("Total Words", f"{total_words:,}")
st.metric("Clean Text Rows", len(nlp_df))
st.metric("Enhanced Rows", len(enhanced_nlp_df))
```

### **5. Upload Section**
Located around **line 994**:
```python
st.header("📁 Upload Financial Slide Deck")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Upload earnings presentations, financial slide decks, or investor presentations"
)
```

### **6. Footer Text**
Located around **line 1442**:
```python
st.markdown("**FinDocPipeline** - Your No-Code ETL Solution for Unstructured Financial Documents")
```

## 🎯 Quick Customization Examples

### **Example 1: Change Brand Colors**
```css
:root {
    --primary-color: #your-brand-color;
    --secondary-color: #your-secondary-color;
    --success-color: #your-success-color;
}

.main-title {
    color: var(--primary-color);
}

.etl-step {
    border-left: 4px solid var(--primary-color);
}
```

### **Example 2: Custom Processing Messages**
Replace the spinner messages with your preferred language:
```python
with st.spinner("🔍 ANALYZING: Scanning your financial presentation..."):
with st.spinner("🧠 PROCESSING: Extracting key financial metrics..."):
with st.spinner("✨ FINALIZING: Preparing your analysis-ready datasets..."):
```

### **Example 3: Custom Metric Labels**
```python
st.metric("📄 Document Pages", len(pages_data))
st.metric("📊 Data Points", f"{total_words:,}")
st.metric("🎯 Clean Records", len(nlp_df))
st.metric("💎 Premium Insights", len(metrics_long_df))
```

## 🔧 Advanced Customization Options

### **1. Custom Logo/Branding**
Add your logo by replacing the emoji in the title:
```python
st.markdown('''
<div style="display: flex; align-items: center; margin-bottom: 2rem;">
    <img src="data:image/png;base64,{your_logo_base64}" width="60" style="margin-right: 1rem;">
    <h1 class="main-title">Your Company Name - Financial ETL Pipeline</h1>
</div>
''', unsafe_allow_html=True)
```

### **2. Custom Color Scheme**
```css
/* Dark theme example */
.stApp {
    background-color: #1e1e1e;
    color: #ffffff;
}

.main-title {
    color: #4fc3f7;
}

.subtitle {
    color: #b0bec5;
}
```

### **3. Custom Fonts**
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
}
```

## 📱 Responsive Design

### **Mobile-Friendly Adjustments**
```css
@media (max-width: 768px) {
    .main-title {
        font-size: 1.8rem;
    }
    
    .subtitle {
        font-size: 1rem;
    }
    
    .metric-card {
        margin-bottom: 1rem;
    }
}
```

## 🚀 Implementation Steps

### **Step 1: Locate the CSS Section**
Open `FinDocPipeline.py` and find the CSS block around line 960.

### **Step 2: Modify Colors/Fonts**
Update the CSS variables and properties to match your brand.

### **Step 3: Update Text Labels**
Search for specific text strings and replace with your preferred language.

### **Step 4: Test Changes**
Restart the Streamlit app to see your changes:
```bash
streamlit run FinDocPipeline.py
```

### **Step 5: Iterate**
Use browser developer tools (F12) to inspect elements and fine-tune styling.

## 🎨 Brand Customization Checklist

- [ ] Update main title and subtitle text
- [ ] Change primary brand colors
- [ ] Customize processing status messages
- [ ] Update metric labels and descriptions
- [ ] Modify success/error message styling
- [ ] Customize download button appearance
- [ ] Update footer branding
- [ ] Add custom logo (optional)
- [ ] Test responsive design on mobile
- [ ] Verify accessibility (color contrast)

## 💡 Pro Tips

1. **Use CSS Variables**: Define colors once and reuse throughout
2. **Test on Different Screens**: Ensure mobile compatibility
3. **Keep Accessibility**: Maintain good color contrast ratios
4. **Brand Consistency**: Match your existing brand guidelines
5. **Performance**: Minimize CSS to keep load times fast

---

**Need Help?** The CSS is located in the main `st.markdown()` block in `FinDocPipeline.py`. All text labels are scattered throughout the file and can be searched/replaced easily.