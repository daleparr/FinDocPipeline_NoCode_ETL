#!/usr/bin/env python3
"""
Debug script to test metrics extraction
"""

import pandas as pd
import re

# Test sample text with financial data
test_text = """
Total Revenue: $125.5 million (increase of 15% year-over-year)
Net Income: $23.8 million (growth of 22% compared to Q4 2022)
Operating Income: $31.2 million
EBITDA: $35.7 million
Total Assets: $450.3 million
Shareholders Equity: $287.6 million
Cash and Cash Equivalents: $67.4 million
Debt-to-Equity Ratio: 0.42
Return on Equity (ROE): 18.5%
Return on Assets (ROA): 12.3%
Earnings Per Share (EPS): $2.15
"""

# Test the current cleaning function
def _clean_text(text):
    """Clean text for NLP processing"""
    if pd.isna(text):
        return ""
    
    # Remove extra whitespace but preserve line breaks initially
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Keep financial symbols and basic punctuation
    # Allow: letters, numbers, spaces, basic punctuation, currency symbols, mathematical symbols
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\%\$£€¥\+\=\|\n\r]', ' ', text)
    
    # Convert line breaks to spaces
    text = re.sub(r'[\n\r]+', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Test metrics patterns
metric_patterns = {
    'revenue': [
        r'(?:total\s+)?revenue[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
        r'(?:net\s+)?sales[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
    ],
    'net_income': [
        r'net\s+income[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?',
        r'net\s+profit[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
    ],
    'total_assets': [
        r'total\s+assets[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
    ],
    'shareholders_equity': [
        r'(?:shareholders?\s+|stockholders?\s+)?equity[:\s]+[\$£€]?([\d,\.]+)\s*(million|billion|m|b|k)?'
    ],
    'eps': [
        r'(?:earnings\s+per\s+share|eps)[:\s]+[\$£€]?([\d,\.]+)'
    ],
    'roe': [
        r'(?:return\s+on\s+equity|roe)[:\s]+([\d,\.]+)%?'
    ]
}

print("=== DEBUGGING METRICS EXTRACTION ===")
print(f"Original text length: {len(test_text)}")
print(f"Original text:\n{test_text[:200]}...")

cleaned_text = _clean_text(test_text)
print(f"\nCleaned text length: {len(cleaned_text)}")
print(f"Cleaned text:\n{cleaned_text[:200]}...")

print("\n=== TESTING PATTERNS ===")
total_matches = 0

for metric_name, patterns in metric_patterns.items():
    print(f"\n{metric_name.upper()}:")
    for pattern in patterns:
        matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
        if matches:
            for match in matches:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else None
                print(f"  + Found: {value} {unit or ''}")
                print(f"    Pattern: {pattern}")
                print(f"    Context: {cleaned_text[max(0, match.start()-20):match.end()+20]}")
                total_matches += 1
        else:
            print(f"  - No match for pattern: {pattern}")

print(f"\nTOTAL MATCHES FOUND: {total_matches}")

if total_matches == 0:
    print("\n=== DEBUGGING INDIVIDUAL PATTERNS ===")
    # Test simple patterns
    simple_patterns = [
        r'revenue[:\s]+\$?([\d,\.]+)',
        r'income[:\s]+\$?([\d,\.]+)',
        r'assets[:\s]+\$?([\d,\.]+)'
    ]
    
    for pattern in simple_patterns:
        matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
        print(f"Pattern '{pattern}': {len(matches)} matches")
        for match in matches:
            print(f"  - {match.group(0)}")