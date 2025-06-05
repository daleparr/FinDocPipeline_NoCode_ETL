# Financial Dictionary Update Guide

## 📋 Overview

This guide provides step-by-step instructions for updating the financial dictionary in your FinDocPipeline system. The financial dictionary controls which terms are recognized, categorized, and extracted from financial documents.

## 🎯 Quick Reference

| Update Type | Code Location | Line Numbers | Restart Required |
|-------------|---------------|--------------|------------------|
| **Basic Financial Terms** | [`NLPDataProcessor._has_financial_terms()`](FinDocPipeline.py:277) | 279-282 | ✅ Yes |
| **NLP Financial Labels** | [`NLPDatasetExporter.__init__()`](FinDocPipeline.py:292) | 293-297 | ✅ Yes |
| **Enhanced Metrics Patterns** | [`DeduplicatedMetricsExtractor.__init__()`](FinDocPipeline.py:806) | 807-846 | ✅ Yes |
| **Basic Financial Patterns** | [`ComprehensiveFinancialParser._extract_financial_data()`](FinDocPipeline.py:155) | 160-177 | ✅ Yes |

## 🔍 Understanding the Financial Dictionary Structure

### 1. Basic Financial Terms (12 terms)
**Purpose**: Quick financial content detection  
**Location**: [`FinDocPipeline.py`](FinDocPipeline.py) lines 279-282

```python
financial_terms = [
    'revenue', 'profit', 'earnings', 'income', 'assets', 'liabilities',
    'equity', 'cash flow', 'margin', 'growth', 'return', 'investment'
]
```

### 2. NLP Financial Labels (18 terms)
**Purpose**: Detailed NLP analysis and entity counting  
**Location**: [`FinDocPipeline.py`](FinDocPipeline.py) lines 293-297

```python
self.financial_labels = [
    'revenue', 'profit', 'loss', 'assets', 'liabilities', 'equity',
    'cash_flow', 'earnings', 'income', 'expenses', 'margin', 'growth',
    'investment', 'debt', 'return', 'dividend', 'share', 'stock'
]
```

### 3. Enhanced Metrics Patterns (12 categories)
**Purpose**: Regex-based financial data extraction  
**Location**: [`FinDocPipeline.py`](FinDocPipeline.py) lines 807-846

## 📝 Step-by-Step Update Procedures

### Procedure 1: Adding Basic Financial Terms

#### When to Use
- Adding new financial concepts for basic detection
- Expanding general financial vocabulary
- Adding industry-specific terms (ESG, crypto, etc.)

#### Steps

1. **Open the file**: [`FinDocPipeline.py`](FinDocPipeline.py)

2. **Navigate to line 279** (or search for `_has_financial_terms`)

3. **Locate the current list**:
```python
financial_terms = [
    'revenue', 'profit', 'earnings', 'income', 'assets', 'liabilities',
    'equity', 'cash flow', 'margin', 'growth', 'return', 'investment'
]
```

4. **Add your new terms**:
```python
# Example: Adding ESG terms
financial_terms = [
    'revenue', 'profit', 'earnings', 'income', 'assets', 'liabilities',
    'equity', 'cash flow', 'margin', 'growth', 'return', 'investment',
    # ESG Terms
    'sustainability', 'carbon footprint', 'esg score', 'governance',
    'environmental impact', 'social responsibility'
]
```

5. **Save the file** (Ctrl+S)

6. **Restart Streamlit** (see [Deployment Guide](DEPLOYMENT_ROLLBACK_GUIDE.md))

#### ✅ Validation
- Upload a document containing your new terms
- Check the "has_financial_terms" field in the output
- Verify new terms are detected

### Procedure 2: Adding NLP Financial Labels

#### When to Use
- Adding terms for detailed NLP analysis
- Creating entity counting for new financial concepts
- Expanding classification capabilities

#### Steps

1. **Navigate to line 293** (or search for `self.financial_labels`)

2. **Locate the current list**:
```python
self.financial_labels = [
    'revenue', 'profit', 'loss', 'assets', 'liabilities', 'equity',
    'cash_flow', 'earnings', 'income', 'expenses', 'margin', 'growth',
    'investment', 'debt', 'return', 'dividend', 'share', 'stock'
]
```

3. **Add your new labels**:
```python
# Example: Adding cryptocurrency terms
self.financial_labels = [
    'revenue', 'profit', 'loss', 'assets', 'liabilities', 'equity',
    'cash_flow', 'earnings', 'income', 'expenses', 'margin', 'growth',
    'investment', 'debt', 'return', 'dividend', 'share', 'stock',
    # Cryptocurrency Terms
    'bitcoin', 'ethereum', 'cryptocurrency', 'blockchain', 'defi',
    'staking', 'mining', 'wallet', 'exchange', 'token'
]
```

4. **Save and restart** (same as Procedure 1)

#### ✅ Validation
- Check NLP dataset output for new `{term}_count` columns
- Verify entity counting works for new terms
- Test financial entity density calculations

### Procedure 3: Adding Enhanced Metrics Patterns

#### When to Use
- Adding new financial metrics with regex extraction
- Improving data extraction accuracy
- Supporting new financial statement formats

#### Steps

1. **Navigate to line 807** (or search for `self.metric_patterns`)

2. **Locate the patterns dictionary**:
```python
self.metric_patterns = {
    'revenue': [
        r'(?:total\s+)?revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
        r'(?:net\s+)?sales[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
    ],
    # ... other patterns
}
```

3. **Add new metric patterns**:
```python
# Example: Adding ESG metrics
self.metric_patterns = {
    # ... existing patterns ...
    'carbon_emissions': [
        r'carbon\s+emissions[:\s]+([\d,\.]+)\s*(tons?|tonnes?|mt|kt)?',
        r'co2\s+emissions[:\s]+([\d,\.]+)\s*(tons?|tonnes?|mt|kt)?'
    ],
    'esg_score': [
        r'esg\s+score[:\s]+([\d,\.]+)',
        r'sustainability\s+rating[:\s]+([\d,\.]+)'
    ],
    'renewable_energy': [
        r'renewable\s+energy[:\s]+([\d,\.]+)%?',
        r'clean\s+energy[:\s]+([\d,\.]+)%?'
    ]
}
```

4. **Save and restart**

#### ✅ Validation
- Test with documents containing new metrics
- Check metrics extraction output
- Verify regex patterns capture data correctly

### Procedure 4: Adding Basic Financial Data Patterns

#### When to Use
- Updating simple revenue/profit extraction patterns
- Modifying basic financial data recognition

#### Steps

1. **Navigate to line 160** (or search for `revenue_patterns`)

2. **Update revenue patterns**:
```python
# Current patterns
revenue_patterns = [
    r'revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?',
    r'total revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?',
    r'net revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b)?'
]

# Enhanced patterns
revenue_patterns = [
    r'revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    r'total revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    r'net revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    # Add new patterns
    r'gross revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    r'operating revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
]
```

3. **Update profit patterns** (lines 173-177):
```python
profit_patterns = [
    r'net income[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    r'profit[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    r'earnings[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    # Add new patterns
    r'operating profit[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
    r'gross profit[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
]
```

## 🎯 Common Update Scenarios

### Scenario A: Adding ESG (Environmental, Social, Governance) Terms

#### Complete Update Checklist
- [ ] **Basic Terms**: Add ESG vocabulary to line 279-282
- [ ] **NLP Labels**: Add ESG terms to line 293-297  
- [ ] **Metrics Patterns**: Add ESG metrics to line 807-846
- [ ] **Test**: Upload ESG report and verify detection

#### Example ESG Terms to Add
```python
# Basic Financial Terms
'sustainability', 'carbon footprint', 'esg score', 'governance',
'environmental impact', 'social responsibility', 'carbon neutral',
'renewable energy', 'diversity', 'board independence'

# NLP Labels (same as above)

# Metrics Patterns
'carbon_emissions': [
    r'carbon\s+emissions[:\s]+([\d,\.]+)\s*(tons?|tonnes?|mt|kt)?'
],
'esg_score': [
    r'esg\s+score[:\s]+([\d,\.]+)'
],
'renewable_percentage': [
    r'renewable\s+energy[:\s]+([\d,\.]+)%?'
]
```

### Scenario B: Adding Cryptocurrency Financial Terms

#### Complete Update Checklist
- [ ] **Basic Terms**: Add crypto vocabulary
- [ ] **NLP Labels**: Add crypto terms for analysis
- [ ] **Metrics Patterns**: Add crypto-specific metrics
- [ ] **Test**: Upload crypto financial report

#### Example Crypto Terms
```python
# Basic Financial Terms
'bitcoin', 'ethereum', 'cryptocurrency', 'blockchain', 'defi',
'staking', 'mining', 'digital assets', 'token', 'wallet'

# Metrics Patterns
'bitcoin_holdings': [
    r'bitcoin\s+holdings[:\s]+([\d,\.]+)\s*(btc)?'
],
'crypto_revenue': [
    r'crypto(?:currency)?\s+revenue[:\s]+[\$£€]?([\d,\.]+)'
],
'staking_rewards': [
    r'staking\s+rewards[:\s]+[\$£€]?([\d,\.]+)'
]
```

### Scenario C: Adding Regulatory/Compliance Terms

#### Example Terms
```python
# Basic Financial Terms
'compliance', 'regulatory capital', 'basel iii', 'stress test',
'capital adequacy', 'liquidity ratio', 'tier 1 capital'

# Metrics Patterns
'tier1_ratio': [
    r'tier\s+1\s+(?:capital\s+)?ratio[:\s]+([\d,\.]+)%?'
],
'liquidity_ratio': [
    r'liquidity\s+(?:coverage\s+)?ratio[:\s]+([\d,\.]+)%?'
]
```

## ⚠️ Important Considerations

### Case Sensitivity
- All terms are automatically converted to lowercase for matching
- Use lowercase in your term lists
- Multi-word terms use spaces (e.g., 'cash flow', not 'cash_flow')

### Regex Pattern Guidelines
- Use `[:\s]+` for flexible separators (colon or space)
- Include currency symbols: `[\$£€]?`
- Support number formats: `[\d,\.]+`
- Include units: `(million|billion|m|b|k)?`
- Use `re.IGNORECASE` flag (automatically applied)

### Performance Impact
- Adding many terms increases processing time
- Complex regex patterns slow down extraction
- Test with large documents to verify performance

## 🧪 Testing Your Updates

### Quick Test Procedure
1. **Save changes** to [`FinDocPipeline.py`](FinDocPipeline.py)
2. **Restart Streamlit** application
3. **Upload test document** containing new terms
4. **Check output** for new term detection
5. **Verify metrics extraction** if applicable

### Detailed Testing
See [Testing & Validation Guide](TESTING_VALIDATION_GUIDE.md) for comprehensive testing procedures.

## 🚨 Troubleshooting

### Common Issues

#### New Terms Not Detected
- **Check spelling** in term lists
- **Verify case sensitivity** (use lowercase)
- **Check document content** actually contains terms
- **Restart Streamlit** application

#### Regex Patterns Not Working
- **Test patterns** using online regex testers
- **Check escaping** of special characters
- **Verify pattern syntax** matches examples
- **Test with simple patterns** first

#### Performance Issues
- **Reduce pattern complexity**
- **Limit number of new terms**
- **Test with smaller documents** first
- **Monitor processing time**

### Error Messages
```python
# Common error: Invalid regex pattern
re.error: bad character range

# Solution: Escape special characters
r'revenue[:\s]+[\$£€]?([\d,\.]+)'  # Correct
r'revenue[:\s]+[$£€]?([\d,\.]+)'   # Incorrect - $ needs escaping
```

## 📚 Related Guides

- **[Master Guide](MASTER_SCHEMA_UPDATE_GUIDE.md)**: Overview and navigation
- **[Pattern Recognition Guide](PATTERN_RECOGNITION_UPDATE_GUIDE.md)**: Advanced regex patterns
- **[Testing Guide](TESTING_VALIDATION_GUIDE.md)**: Validation procedures
- **[Deployment Guide](DEPLOYMENT_ROLLBACK_GUIDE.md)**: Production updates

---

## 📝 Version Control

**Last Updated**: January 6, 2025  
**Compatible with**: FinDocPipeline v2.0.0  
**Guide Version**: 1.0.0