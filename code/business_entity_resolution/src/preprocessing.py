#!/usr/bin/env python3
"""
preprocessing.py
Text cleaning, entity normalization, and legal suffix stripping.
Zero external lookups. Handles US, India, France, and any unseen open country set.
"""

import re
import unicodedata
from typing import Optional

# Common business suffixes across international markets (US, UK, India, France, etc.)
LEGAL_SUFFIXES = [
    r"\bpvt\b", r"\bltd\b", r"\bpvt\s+ltd\b", r"\bprivate\s+limited\b",
    r"\binc\b", r"\bincorporated\b", r"\bcorp\b", r"\bcorporation\b",
    r"\bllc\b", r"\bllp\b", r"\bco\b", r"\bcompany\b",
    r"\bsa\b", r"\bsarl\b", r"\bsas\b", r"\beurl\b", r"\bgmbh\b",
    r"\benterprises?\b", r"\bservices?\b", r"\bsolutions?\b", r"\btechnologies?\b"
]
LEGAL_SUFFIX_RE = re.compile(r"|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

# Common address abbreviations
ADDRESS_ABBRS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bno\b": "number",
    r"\bopp\b": "opposite",
    r"\bnear\b": "near",
    r"\bbldg\b": "building",
    r"\bsec\b": "sector",
    r"\bstr\b": "strasse",
    r"\brue\b": "rue",
    r"\bav\b": "avenue",
    r"\bbvd\b": "boulevard"
}

def unicode_normalize(text: Optional[str]) -> str:
    """Normalize unicode, strip accents (useful for French names/addresses)."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize unicode (decompose accented chars like é -> e + accent)
    text = unicodedata.normalize("NFKD", text)
    # Strip non-spacing marks (accents)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()

def clean_business_name(name: Optional[str], strip_legal: bool = True) -> str:
    """
    Standardize business names:
    - Unicode & lowercase
    - Replace '&' with 'and'
    - Remove punctuation and extra whitespace
    - Optionally remove legal entity suffixes
    """
    text = unicode_normalize(name)
    if not text:
        return ""
    
    # Replace & and + with 'and'
    text = text.replace("&", " and ").replace("+", " and ")
    
    if strip_legal:
        text = LEGAL_SUFFIX_RE.sub(" ", text)

    # Remove all punctuation except alphanumeric
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_address(address: Optional[str]) -> str:
    """
    Standardize business addresses:
    - Expand standard road/street abbreviations
    - Normalize delimiters
    - Keep digits (crucial for PIN codes, suite numbers)
    """
    text = unicode_normalize(address)
    if not text:
        return ""

    # Replace punctuation with spaces
    text = re.sub(r"[,;:\-\.\/]", " ", text)

    # Expand common abbreviations
    for pattern, replacement in ADDRESS_ABBRS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_digits(text: Optional[str]) -> list[str]:
    """Extract digit sequences (PIN codes, zip codes, building numbers)."""
    if not text:
        return []
    return re.findall(r"\b\d{2,8}\b", str(text))

def clean_country(country: Optional[str]) -> str:
    """Standardize country string."""
    if not country or not isinstance(country, str):
        return "unknown"
    c = unicode_normalize(country).strip()
    if c in ("us", "usa", "united states", "united states of america"):
        return "us"
    if c in ("in", "ind", "india", "bharat"):
        return "india"
    if c in ("fr", "fra", "france"):
        return "france"
    return c
