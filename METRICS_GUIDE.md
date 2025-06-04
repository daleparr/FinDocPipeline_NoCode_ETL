# FinDocPipeline - Dashboard Metrics & ETL Pipeline Guide

## 📊 Dashboard Metrics Explained

### ETL Pipeline Summary Dashboard
When you process a financial slide deck, the top dashboard shows 7 key metrics:

| Metric | What It Means | Example | Good/Bad Indicator |
|--------|---------------|---------|-------------------|
| **Slides Processed** | Total PDF slides analyzed | 54 | More slides = more comprehensive analysis |
| **Total Words** | All text extracted and processed | 21,576 | Higher count = more content captured |
| **Clean Text Rows** | Text segments ready for NLP analysis | 77 | Should match or exceed slides processed |
| **Enhanced Rows** | Segments including visual content | 77 | Same or higher than Clean Text |
| **NLP Dataset Rows** | Final dataset with features/labels | 77 | Final processed dataset size |
| **Extracted Metrics** | Financial ratios found by METRIC_PATTERNS | 18 | More metrics = better extraction success |
| **Tables/Charts** | Format: `{tables}/{images}` | 34/0 | See detailed explanation below |

## 🔍 Tables/Charts Metric Deep Dive

### Format: `{Tables}/{Images}`

**Example: 34/0**
- **34 Tables**: Found 34 structured data tables (financial statements, data grids)
- **0 Images**: Found 0 visual elements (charts, graphs, diagrams)

### What Tables Include
- Financial statements (Income Statement, Balance Sheet, Cash Flow)
- Banking ratio tables (CET1, NPL, LCR summaries)
- Performance metrics tables (ROE, ROA, NIM)
- Regulatory capital tables
- Structured lists with consistent formatting

### What Images/Charts Include
- Bar charts, line graphs, pie charts
- Performance trend visualizations
- Embedded images and diagrams
- Infographics and illustrations
- Visual KPI dashboards

### Interpreting Different Patterns

| Pattern | Meaning | Document Type | ETL Success Indicator |
|---------|---------|---------------|----------------------|
| **High/Low** (34/0) | Table-heavy slide deck | Earnings presentations, regulatory filings | ✅ Excellent for METRIC_PATTERNS extraction |
| **Low/High** (5/15) | Visual-heavy slide deck | Investor presentations, marketing materials | ⚠️ May need OCR review for chart data |
| **High/High** (25/12) | Comprehensive slide deck | Annual reports, detailed presentations | ✅ Rich content, optimal ETL potential |
| **Low/Low** (2/1) | Minimal structured content | Press releases, narrative reports | ⚠️ Limited metrics extraction expected |

## 📈 METRIC_PATTERNS Confidence Guide

### Understanding Confidence Levels

| Score Range | Quality Level | Meaning | Recommended Action |
|-------------|---------------|---------|-------------------|
| **0.8 - 1.0** | High | Found in structured format with clear context | ✅ Use directly in analysis |
| **0.6 - 0.8** | Medium | Found but may need verification | ⚠️ Review manually before use |
| **0.5 - 0.6** | Low | Found but uncertain context | 🔍 Verify against source slides |
| **< 0.5** | Rejected | Not included in final datasets | ❌ Not reliable for analysis |

### Confidence Boosters in METRIC_PATTERNS
- **+0.2**: Found in a table or structured format
- **+0.2**: Found in financial statement context
- **+0.1**: Has currency symbol ($, £, €, ¥) or percentage (%)
- **Base**: 0.5 starting confidence for any pattern match

### Example Confidence Calculations
```
"CET1 ratio: 13.4%" in a regulatory table
= 0.5 (base) + 0.2 (table) + 0.2 (financial context) + 0.1 (percentage) = 1.0

"ROE 12.8%" in narrative text  
= 0.5 (base) + 0.2 (financial context) + 0.1 (percentage) = 0.8

"125.5 million" without context
= 0.5 (base) = 0.5 (likely rejected)
```

## 🎯 ETL Pipeline Quality Indicators

### Extraction Success Indicators
- **High Table Count**: More structured data available for METRIC_PATTERNS
- **Multiple Metrics Found**: Comprehensive regex pattern matching
- **High Average Confidence**: Quality extractions from clean text
- **Context Preservation**: Full audit trail for validation

### Warning Signs
- **Zero Metrics Extracted**: Slide deck may be image-based or poorly formatted
- **Low Confidence Scores**: METRIC_PATTERNS may need manual verification
- **Processing Errors**: Check slide deck quality and format
- **Mismatched Expectations**: Document type may not match banking/financial assumptions

## 📋 Slide Deck Type Expectations

### Earnings Presentations
- **Expected Tables**: 20-50
- **Expected Images**: 5-15  
- **Expected Metrics**: 15-30 (CET1, ROE, NPL, etc.)
- **Processing Time**: 3-8 minutes

### Regulatory Filings (Pillar 3)
- **Expected Tables**: 30-60
- **Expected Images**: 2-8
- **Expected Metrics**: 20-40 (Capital ratios, risk metrics)
- **Processing Time**: 5-12 minutes

### Investor Presentations
- **Expected Tables**: 15-40
- **Expected Images**: 10-25
- **Expected Metrics**: 12-25 (Performance KPIs)
- **Processing Time**: 2-6 minutes

### Quarterly Updates
- **Expected Tables**: 10-25
- **Expected Images**: 5-15
- **Expected Metrics**: 10-20 (Key ratios and trends)
- **Processing Time**: 2-5 minutes

## 🔧 ETL Pipeline Troubleshooting

### If Extracted Metrics = 0
1. **Check Slide Deck Type**: Is it a financial presentation with numerical data?
2. **Check Text Quality**: Can you select/copy text from the PDF slides?
3. **Review Clean Text**: Look at text samples in the clean dataset
4. **Document Format**: Native PDF vs scanned image slides?

### If Tables = 0
1. **Slide Structure**: Does it have structured data tables?
2. **PDF Quality**: Is the table structure preserved in extraction?
3. **Format Issues**: Tables might be formatted as images rather than text

### If Confidence Scores Low
1. **Review Context**: Check surrounding text for each extracted metric
2. **Slide Quality**: Poor formatting affects METRIC_PATTERNS confidence
3. **Manual Verification**: Cross-check against source slides

### If ETL Pipeline Fails
1. **File Size**: Keep slide decks under 100MB for optimal performance
2. **File Format**: Ensure it's a valid PDF file with selectable text
3. **Permissions**: Check if PDF is password-protected
4. **Memory**: Large slide decks may need more processing time

## 📊 ETL Performance Benchmarks

### Typical Results by Slide Count

| Slides | Processing Time | Expected Tables | Expected Metrics | Memory Usage |
|--------|----------------|-----------------|------------------|--------------|
| 1-10 | 15-30 seconds | 2-8 | 3-8 | <50MB |
| 11-25 | 30-90 seconds | 5-15 | 6-15 | 50-100MB |
| 26-50 | 1-3 minutes | 10-25 | 10-20 | 100-200MB |
| 51-100 | 2-5 minutes | 20-40 | 15-30 | 200-400MB |
| 100+ | 5-10 minutes | 30+ | 20+ | 400MB+ |

## 🎯 Output Dataset Guide

### Clean Text Dataset (`cleaned_for_nlp.csv`)
- **Purpose**: NLP-ready text with normalized tokens
- **Key Features**: `<USD>`, `<PCT>` token normalization, theme tagging
- **Use Cases**: Sentiment analysis, topic modeling, text classification

### Structured Metrics (Long-Form: `metrics_extracted.csv`)
- **Purpose**: One row per metric occurrence with full context
- **Key Features**: Confidence scores, source slide tracking, pattern attribution
- **Use Cases**: Time-series analysis, trend tracking, audit trails

### Structured Metrics (Wide-Form: `metrics_extracted_wide.csv`)
- **Purpose**: One row per slide with metrics as columns
- **Key Features**: Excel-friendly format, pivot-ready structure
- **Use Cases**: Comparative analysis, financial modeling, dashboards

### JSON Export (`metrics_extracted.json`)
- **Purpose**: API-ready structured data
- **Key Features**: Nested structure, metadata preservation
- **Use Cases**: Database ingestion, system integration, programmatic access

---

**ETL Success Tip**: The Tables/Charts metric (e.g., 34/0) is often the best predictor of METRIC_PATTERNS extraction success. Slide decks with high table counts typically yield the most comprehensive financial metrics through the regex library.