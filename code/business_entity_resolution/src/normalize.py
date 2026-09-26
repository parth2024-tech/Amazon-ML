#!/usr/bin/env python3
"""
normalize.py
Comprehensive text normalization, cleaning, and token preservation.
Handles international records (US, India, France, and any open country set).
"""

import re
import unicodedata
from typing import List, Optional
from anyascii import anyascii

# Legal entity suffixes across US, India, UK, France, Germany, etc.
LEGAL_SUFFIXES = [
    r"\bpublic\s+limited\s+company\b",
    r"\bprivate\s+limited\b",
    r"\bpvt\s+ltd\b",
    r"\bltd\b",
    r"\bpvt\b",
    r"\binc\b",
    r"\bincorporated\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\bsa\b",
    r"\bsarl\b",
    r"\bsas\b",
    r"\beurl\b",
    r"\bgmbh\b",
    r"\benterprises?\b",
    r"\bservices?\b",
    r"\bsolutions?\b",
    r"\btechnologies?\b",
    r"\bconsultants?\b",
    r"\bconsulting\b",
    r"\bcenters?\b",
    r"\bcentres?\b",
    r"\bassociates\b",
    r"\bholdings\b",
    r"\bgroups?\b",
    # Transliterated Indic legal forms
    r"\bpiraivet\s+limitet\b",
    r"\bpraivet\s+limited\b",
    r"\bpraivet\s+limitet\b",
    r"\belelp\b",
    r"\blimitet\b",
    r"\bpiraivet\b",
    r"\bpraivet\b"
]
LEGAL_SUFFIX_RE = re.compile(r"|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)
DOMAIN_RE = re.compile(r"\.(com|net|org|in|co\.in|io|biz|info|edu|gov|us|fr)\b", flags=re.IGNORECASE)

# Standard address abbreviation mappings
ADDRESS_ABBREVIATIONS = {
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

def unicode_normalize(text: str | None) -> str:
    """Normalize unicode, transliterate non-Latin scripts, decompose and strip accents (é -> e, etc.)."""
    if not text or not isinstance(text, str):
        return ""
    text = anyascii(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()

def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()

def normalize_business_name(name: str | None, strip_legal: bool = True) -> str:
    """
    Standardize business names:
    - Unicode & lowercase
    - Strip web domain extensions (.com, .net, etc.)
    - Replace '&' and '+' with 'and'
    - Normalize punctuation delimiters to spaces
    - Collapse single-letter acronym sequences (l l c -> llc)
    - Remove/expand legal entity suffixes
    - Remove special punctuation while preserving alphanumerics (including Unicode)
    """
    text = unicode_normalize(name)
    if not text:
        return ""

    text = DOMAIN_RE.sub(" ", text)
    text = text.replace("&", " and ").replace("+", " and ")

    # Punctuation to spaces
    text = re.sub(r"[.,\-–—_/\\()\"'’`#@:;!*\[\]{}~?]", " ", text)
    # Collapse single-letter acronyms: "l l c" -> "llc", "d b a" -> "dba"
    text = re.sub(r"\b([a-z])\s+([a-z])\s+([a-z])\b", r"\1\2\3", text)
    text = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", text)

    if strip_legal:
        text = LEGAL_SUFFIX_RE.sub(" ", text)

    # Preserve alphanumeric characters across all scripts (Latin, Devanagari, Tamil, etc.)
    text = "".join(c for c in text if c.isalnum() or c.isspace())
    return normalize_whitespace(text)

def normalize_address(address: str | None) -> str:
    """
    Standardize business addresses:
    - Unicode & lowercase
    - Expand standard street/suite abbreviations
    - Replace punctuation with delimiters
    - Preserve all numeric tokens (PIN codes, ZIP codes, door numbers)
    """
    text = unicode_normalize(address)
    if not text:
        return ""

    text = re.sub(r"[,;:\-\.\/]", " ", text)

    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text)

def extract_geographic_tokens(address: str | None) -> list[str]:
    """
    Extract postal codes (US 5-digit, India 6-digit, France 5-digit),
    and building/plot numbers. Normalizes numbers by removing leading zeros.
    """
    if not address:
        return []
    raw = re.findall(r"\d+", str(address))
    tokens: list[str] = []
    for t in raw:
        clean = t.lstrip("0")
        if clean and len(clean) <= 8 and clean not in tokens:
            tokens.append(clean)
    return tokens

def normalize_country(country: str | None) -> str:
    """Standardize country label without restricting to a closed set."""
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

