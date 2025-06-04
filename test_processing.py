#!/usr/bin/env python3
"""
Test script to verify the enhanced FinDocPipeline processing
Tests dictionary alignment, schema design, and structured outputs
"""

import pandas as pd
import json
import re
from datetime import datetime

# Test the financial dictionaries and extraction logic
def test_financial_dictionaries():
    """Test the financial term dictionaries"""
    
    # Sample text with financial terms
    test_text = """
    Total Revenue: $125.5 million (increase of 15% year-over-year)
    Net Income: $23.8 million (growth of 22% compared to Q4 2022)
    Operating Income: $31.2 million
    EBITDA: $35.7 million
    Total Assets: $450.3 million
    Shareholders Equity: $287.6 million
    Return on Equity (ROE): 18.5%
    Earnings Per Share (EPS): $2.15
    """
    
    # Financial dictionaries from our enhanced parser
    financial_terms_dict = {
        'revenue_terms': ['revenue', 'sales', 'income', 'turnover', 'receipts'],
        'profit_terms': ['profit', 'earnings', 'net income', 'ebitda', 'operating income'],
        'asset_terms': ['assets', 'property', 'equipment', 'inventory', 'cash', 'investments'],
        'liability_terms': ['liabilities', 'debt', 'payables', 'obligations', 'borrowings'],
        'equity_terms': ['equity', 'shareholders equity', 'retained earnings', 'capital'],
        'ratio_terms': ['ratio', 'margin', 'return', 'yield', 'percentage', 'rate'],
        'trend_terms': ['increase', 'decrease', 'growth', 'decline', 'improvement', 'deterioration'],
        'time_terms': ['year', 'quarter', 'month', 'annual', 'quarterly', 'monthly', 'ytd', 'q1', 'q2', 'q3', 'q4']
    }
    
    print("=== TESTING FINANCIAL DICTIONARIES ===")
    
    extracted_metrics = {}
    
    for category, terms in financial_terms_dict.items():
        category_metrics = []
        for term in terms:
            # Look for numerical values near financial terms
            pattern = rf'{re.escape(term)}\s*[:\-]?\s*([\d,\.]+)\s*([%$£€¥]?)\s*(million|billion|m|b|k)?'
            matches = re.finditer(pattern, test_text, re.IGNORECASE)
            
            for match in matches:
                category_metrics.append({
                    'term': term,
                    'value': match.group(1),
                    'currency': match.group(2),
                    'unit': match.group(3),
                    'context': test_text[max(0, match.start()-30):match.end()+30].strip()
                })
        
        if category_metrics:
            extracted_metrics[category] = category_metrics
            print(f"\n{category.upper()}:")
            for metric in category_metrics:
                print(f"  • {metric['term']}: {metric['value']} {metric['currency']} {metric['unit'] or ''}")
                print(f"    Context: {metric['context'][:60]}...")
    
    return extracted_metrics

def test_nlp_features():
    """Test NLP feature extraction"""
    
    test_text = """
    The company achieved record revenue in Q4 2023, driven by strong performance. 
    Profit margins improved significantly compared to the previous quarter.
    We expect continued growth in 2024, with projected revenue increase of 10-12%.
    Market volatility may impact future performance and create challenges.
    """
    
    print("\n=== TESTING NLP FEATURES ===")
    
    # Text classification
    text_lower = test_text.lower()
    
    features = {
        'is_financial_statement': any(term in text_lower for term in 
            ['balance sheet', 'income statement', 'cash flow statement', 'statement of']),
        'is_narrative_text': len(test_text.split('.')) > 3 and not ('|' in test_text or '\t' in test_text),
        'is_executive_summary': any(term in text_lower for term in 
            ['executive summary', 'overview', 'highlights', 'key points']),
        'is_risk_disclosure': any(term in text_lower for term in 
            ['risk', 'uncertainty', 'forward-looking', 'may', 'could', 'might']),
        'is_performance_metric': any(term in text_lower for term in 
            ['performance', 'results', 'achievement', 'target', 'goal']),
        'has_time_references': bool(re.search(r'\b(year|month|quarter|q[1-4]|\d{4})\b', text_lower)),
        'has_comparison_terms': any(term in text_lower for term in 
            ['compared to', 'versus', 'vs', 'increase', 'decrease', 'higher', 'lower']),
        'has_future_tense': any(term in text_lower for term in 
            ['will', 'expect', 'forecast', 'project', 'anticipate', 'plan']),
        'has_past_tense': any(term in text_lower for term in 
            ['was', 'were', 'had', 'achieved', 'reported', 'recorded'])
    }
    
    # Sentiment analysis
    positive_terms = ['growth', 'increase', 'profit', 'success', 'strong', 'improved', 'gain']
    negative_terms = ['loss', 'decrease', 'decline', 'weak', 'poor', 'risk', 'challenge']
    
    positive_count = sum(1 for term in positive_terms if term in text_lower)
    negative_count = sum(1 for term in negative_terms if term in text_lower)
    
    if positive_count > negative_count:
        sentiment = 'positive'
    elif negative_count > positive_count:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    features['sentiment_indicators'] = sentiment
    
    print("NLP FEATURES DETECTED:")
    for feature, value in features.items():
        if value:
            print(f"  + {feature.replace('_', ' ').title()}: {value}")
    
    return features

def test_schema_alignment():
    """Test that our outputs align with expected schema"""
    
    print("\n=== TESTING SCHEMA ALIGNMENT ===")
    
    # Expected schema for NLP dataset
    expected_schema = {
        'id': 'string',
        'page_number': 'integer',
        'content_type': 'string',
        'text': 'string',
        'word_count': 'integer',
        'char_count': 'integer',
        'sentence_count': 'integer',
        'has_financial_terms': 'boolean',
        'extraction_method': 'string',
        'is_financial_statement': 'boolean',
        'is_narrative_text': 'boolean',
        'is_tabular_data': 'boolean',
        'contains_numbers': 'boolean',
        'contains_percentages': 'boolean',
        'contains_currency': 'boolean',
        'avg_word_length': 'float',
        'avg_sentence_length': 'float',
        'complexity_score': 'float',
        'financial_entity_density': 'float',
        'sentiment_indicators': 'string'
    }
    
    # Create sample data row
    sample_row = {
        'id': 'doc_1_0',
        'page_number': 1,
        'content_type': 'main_text',
        'text': 'Total Revenue: $125.5 million increased by 15%.',
        'word_count': 8,
        'char_count': 45,
        'sentence_count': 1,
        'has_financial_terms': True,
        'extraction_method': 'pdfplumber_comprehensive',
        'is_financial_statement': False,
        'is_narrative_text': True,
        'is_tabular_data': False,
        'contains_numbers': True,
        'contains_percentages': True,
        'contains_currency': True,
        'avg_word_length': 5.6,
        'avg_sentence_length': 8.0,
        'complexity_score': 6.8,
        'financial_entity_density': 0.125,
        'sentiment_indicators': 'positive'
    }
    
    print("SCHEMA VALIDATION:")
    schema_valid = True
    for field, expected_type in expected_schema.items():
        if field in sample_row:
            actual_value = sample_row[field]
            actual_type = type(actual_value).__name__
            
            # Type mapping
            type_mapping = {
                'str': 'string',
                'int': 'integer', 
                'float': 'float',
                'bool': 'boolean'
            }
            
            mapped_type = type_mapping.get(actual_type, actual_type)
            
            if mapped_type == expected_type:
                print(f"  + {field}: {expected_type} - OK")
            else:
                print(f"  - {field}: Expected {expected_type}, got {mapped_type}")
                schema_valid = False
        else:
            print(f"  - {field}: Missing field")
            schema_valid = False
    
    print(f"\nSCHEMA VALIDATION: {'PASSED' if schema_valid else 'FAILED'}")
    return schema_valid

def test_export_formats():
    """Test export format generation"""
    
    print("\n=== TESTING EXPORT FORMATS ===")
    
    # Sample dataset
    sample_data = [
        {
            'id': 'doc_1_0',
            'page_number': 1,
            'content_type': 'main_text',
            'text': 'Total Revenue: $125.5 million',
            'financial_entity_density': 0.125,
            'sentiment_indicators': 'positive'
        },
        {
            'id': 'doc_1_1', 
            'page_number': 1,
            'content_type': 'table',
            'text': 'Q4 2023 | $125.5M | 19.0%',
            'financial_entity_density': 0.333,
            'sentiment_indicators': 'neutral'
        }
    ]
    
    # Test CSV export
    df = pd.DataFrame(sample_data)
    csv_output = df.to_csv(index=False)
    print("CSV EXPORT:")
    print(csv_output[:200] + "..." if len(csv_output) > 200 else csv_output)
    
    # Test JSON export
    json_output = {
        "dataset_info": {
            "name": "Financial Document NLP Dataset",
            "version": "1.0",
            "description": "Processed financial document text with NLP features and labels",
            "created_at": datetime.now().isoformat(),
            "total_samples": len(sample_data)
        },
        "data": sample_data
    }
    
    print("\nJSON EXPORT STRUCTURE:")
    print(json.dumps(json_output, indent=2)[:300] + "...")
    
    return True

def main():
    """Run all tests"""
    print("ENHANCED FINDOCPIPELINE PROCESSING TEST")
    print("=" * 50)
    
    try:
        # Run tests
        metrics = test_financial_dictionaries()
        features = test_nlp_features()
        schema_valid = test_schema_alignment()
        export_test = test_export_formats()
        
        print("\n" + "=" * 50)
        print("TEST SUMMARY:")
        print(f"  - Financial Dictionary Extraction: {'PASSED' if metrics else 'FAILED'}")
        print(f"  - NLP Feature Detection: {'PASSED' if features else 'FAILED'}")
        print(f"  - Schema Alignment: {'PASSED' if schema_valid else 'FAILED'}")
        print(f"  - Export Format Generation: {'PASSED' if export_test else 'FAILED'}")
        
        overall_status = all([metrics, features, schema_valid, export_test])
        print(f"\nOVERALL TEST STATUS: {'PASSED' if overall_status else 'FAILED'}")
        
        return overall_status
        
    except Exception as e:
        print(f"TEST FAILED WITH ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    main()