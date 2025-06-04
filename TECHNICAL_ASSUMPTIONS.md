# FinDocPipeline - Technical Assumptions & Business Logic

## 🔧 Core Technical Assumptions

### PDF Processing
- **Text Extraction Priority**: pdfplumber preferred over PyMuPDF for better table detection
- **Encoding**: UTF-8 encoding assumed for all text content
- **Page Processing**: Sequential page-by-page processing to maintain document structure
- **Memory Management**: Temporary files cleaned up after processing to prevent memory leaks

### Text Cleaning & Normalization
- **Financial Symbol Preservation**: Currency symbols ($, £, €, ¥) and mathematical operators (+, =, %, |) are preserved
- **Whitespace Normalization**: Multiple spaces/tabs collapsed to single spaces, line breaks converted to spaces
- **Character Filtering**: Non-alphanumeric characters removed except for preserved financial symbols
- **Minimum Content Length**: Text segments under 10 characters filtered out as noise

### Pattern Matching Logic
- **Case Insensitive**: All regex patterns use `re.IGNORECASE` flag
- **Multiple Patterns**: 2-3 regex patterns per metric to handle format variations
- **Bidirectional Matching**: Patterns match both "Revenue: $125M" and "$125M Revenue" formats
- **Number Format Flexibility**: Handles comma separators, decimal points, and unit suffixes (million, billion, k, m, b)

## 📊 Metrics Extraction Business Logic

### Supported Financial Metrics
```python
METRICS = {
    'revenue': ['total revenue', 'net sales', 'sales revenue'],
    'net_income': ['net income', 'net profit', 'net earnings'],
    'total_assets': ['total assets'],
    'shareholders_equity': ['shareholders equity', 'stockholders equity', 'equity'],
    'cash_and_equivalents': ['cash and cash equivalents', 'cash and equivalents'],
    'operating_income': ['operating income', 'operating profit'],
    'gross_profit': ['gross profit', 'gross income'],
    'ebitda': ['ebitda'],
    'eps': ['earnings per share', 'eps'],
    'roe': ['return on equity', 'roe'],
    'roa': ['return on assets', 'roa'],
    'debt_to_equity': ['debt to equity', 'debt-to-equity']
}
```

### Confidence Scoring Algorithm
```python
def calculate_confidence(text, match, metric_name):
    confidence = 0.5  # Base confidence
    
    # Table context bonus
    if 'table' in text.lower() or '|' in text:
        confidence += 0.2
    
    # Financial statement context bonus
    financial_contexts = ['balance sheet', 'income statement', 'cash flow', 'financial']
    if any(ctx in text.lower() for ctx in financial_contexts):
        confidence += 0.2
    
    # Currency symbol bonus
    if any(symbol in match.group(0) for symbol in ['$', '£', '€', '¥']):
        confidence += 0.1
    
    return min(confidence, 1.0)  # Cap at 1.0
```

### Deduplication Strategy
- **Grouping Key**: `(page_number, metric_name)` - allows same metric on different pages
- **Selection Criteria**: Highest confidence score within each group
- **Tie Breaking**: First occurrence selected if confidence scores are equal
- **Context Preservation**: Original context and pattern information retained for audit

## 🖼️ Computer Vision & OCR Logic

### Image Detection
- **Supported Formats**: PNG, JPEG, embedded PDF images
- **Size Filtering**: Images smaller than 50x50 pixels ignored as decorative elements
- **Color Analysis**: Basic color distribution analysis for chart type detection
- **Chart Type Indicators**: Pattern matching on OCR text for chart type keywords

### OCR Processing
- **Engine**: Tesseract OCR with default configuration
- **Text Cleaning**: Common OCR errors corrected (|l1 → I, O0 → 0)
- **Noise Filtering**: Non-alphanumeric characters except financial symbols removed
- **Context Integration**: OCR text integrated with main document text for metrics extraction

### Visual Content Classification
```python
CHART_INDICATORS = {
    'bar_chart': ['bar chart', 'bar graph', 'column chart', 'histogram'],
    'line_chart': ['line chart', 'line graph', 'trend line', 'time series'],
    'pie_chart': ['pie chart', 'pie graph', 'donut chart', 'circular chart'],
    'scatter_plot': ['scatter plot', 'scatter chart', 'correlation plot'],
    'table': ['table', 'matrix', 'grid', 'tabular data']
}
```

## 📈 NLP Dataset Creation Logic

### Text Classification Labels
- **is_financial_statement**: Contains balance sheet, income statement, or cash flow keywords
- **is_narrative_text**: Paragraph-style text without tabular structure
- **is_tabular_data**: Contains table indicators (|, multiple columns, structured layout)
- **is_executive_summary**: Contains summary, overview, or highlights keywords
- **is_risk_disclosure**: Contains risk, uncertainty, or forward-looking statement keywords
- **is_performance_metric**: Contains financial metrics or KPI keywords

### Feature Engineering
- **Text Statistics**: Word count, character count, sentence count
- **Readability Metrics**: Average word length, sentence length, complexity score
- **Financial Entity Density**: Ratio of financial terms to total words
- **Temporal References**: Presence of time-related terms (quarterly, annual, YTD)
- **Sentiment Indicators**: Basic positive/negative financial sentiment

### Quality Validation
- **Range Checking**: Financial metrics validated against reasonable ranges
- **Format Validation**: Number formats checked for consistency
- **Context Validation**: Surrounding text analyzed for relevance
- **Cross-Reference**: Metrics compared across pages for consistency

## 🔄 Processing Pipeline Assumptions

### Error Handling
- **Graceful Degradation**: Processing continues even if individual pages fail
- **Fallback Methods**: PyMuPDF used if pdfplumber fails
- **Error Logging**: All exceptions logged with context for debugging
- **Partial Results**: Successful extractions returned even if some pages fail

### Performance Optimization
- **Memory Management**: Large documents processed in chunks
- **Caching**: Compiled regex patterns cached for reuse
- **Parallel Processing**: Page-level processing could be parallelized (not currently implemented)
- **Resource Limits**: Processing timeout after 10 minutes for very large documents

### Data Integrity
- **Immutable Source**: Original PDF never modified
- **Audit Trail**: Complete processing history maintained
- **Version Control**: Extraction patterns versioned for reproducibility
- **Validation**: All extracted data validated before export

## 🎯 Business Rule Assumptions

### Document Types
- **Primary**: Annual reports, quarterly reports, earnings presentations
- **Secondary**: Financial statements, regulatory filings, investor presentations
- **Excluded**: Scanned documents without selectable text, non-financial documents

### Metric Prioritization
1. **Revenue Metrics**: Highest priority for business analysis
2. **Profitability Metrics**: Net income, operating income, gross profit
3. **Balance Sheet Items**: Assets, equity, cash positions
4. **Financial Ratios**: ROE, ROA, leverage ratios
5. **Per-Share Metrics**: EPS, book value per share

### Quality Thresholds
- **High Confidence**: ≥0.8 - Ready for automated processing
- **Medium Confidence**: 0.6-0.8 - Suitable with human review
- **Low Confidence**: 0.5-0.6 - Requires manual verification
- **Rejected**: <0.5 - Not included in final results

## 🔍 Validation & Testing Assumptions

### Test Data Requirements
- **Diverse Formats**: Multiple document layouts and styles
- **Known Ground Truth**: Manually verified correct values for comparison
- **Edge Cases**: Unusual formatting, missing data, corrupted files
- **Scale Testing**: Documents ranging from 1-1000+ pages

### Accuracy Metrics
- **Precision**: Percentage of extracted metrics that are correct
- **Recall**: Percentage of actual metrics that were found
- **F1 Score**: Harmonic mean of precision and recall
- **Confidence Calibration**: Alignment between confidence scores and actual accuracy

### Performance Benchmarks
- **Processing Speed**: Target <5 seconds per page
- **Memory Usage**: Target <1GB for 100-page documents
- **Accuracy**: Target >90% for standard financial documents
- **Reliability**: Target <1% processing failures

---

**Note**: These assumptions are based on common financial document formats and may need adjustment for specialized or non-standard documents. Regular validation against new document types is recommended to maintain accuracy.