"""
========================================================================================
DocGuard: Multimodal Document Forgery & Manipulation Detection Platform
Module 9: Deterministic Arithmetic Logic Verifier (logic_checker.py)
----------------------------------------------------------------------------------------
Purpose:
  Provides a symbolic, deterministic verification engine that extracts line-item
  amounts and mathematical totals from financial documents, auditing whether the
  underlying arithmetic ledger balances:
      sum(Line Items) [+ Tax] [- Discount] == Declared Grand Total

Neuro-Symbolic Fraud Detection Rationale:
  Neural deep learning models (such as ResNet-50 and EfficientNet-B0) excel at visual
  feature representation, detecting high-frequency ELA noise, edge seams, and
  compression discontinuities. However, neural networks are statistical classifiers
  susceptible to hallucination, boundary drift, and out-of-distribution errors.

  Conversely, deterministic symbolic rule engines are 100% rigorous, explainable, and
  non-hallucinatory: an altered monetary total (e.g. changing 4,000 to 14,000 while
  leaving 1,000 and 3,000 line items untouched) creates an immediate mathematical
  contradiction:
      1,000 + 3,000 = 4,000 != 14,000 (Discrepancy: +10,000)

  Combining neural visual analysis with deterministic symbolic ledger checking produces
  a fail-safe defense against sophisticated invoice tampering.
========================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Optional PyMuPDF integration for direct PDF text extraction
try:
    import pymupdf  # modern PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # legacy PyMuPDF import
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False


# =============================================================================
# Regular Expression Patterns for Financial Invoices
# =============================================================================

# Currency symbols and abbreviations
CURRENCY_PATTERN = r"(?:INR|Rs\.?|₹|USD|\$|EUR|€|GBP|£|CAD|AUD|JPY|¥)"

# Number pattern matching Western (1,000.00) and South Asian (1,00,000.00) commas
NUMBER_PATTERN = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?"

# Compiled patterns
RE_CURRENCY = re.compile(CURRENCY_PATTERN, re.IGNORECASE)
RE_AMOUNT = re.compile(
    rf"(?:({CURRENCY_PATTERN})\s*)?(-?\s*(?:{NUMBER_PATTERN}))",
    re.IGNORECASE,
)

# Header / Total line indicators
RE_GRAND_TOTAL = re.compile(
    r"\b(?:grand\s+total|total\s+amount|net\s+payable|amount\s+due|balance\s+due|final\s+total|total)\b",
    re.IGNORECASE,
)

RE_SUBTOTAL = re.compile(
    r"\b(?:sub\s*total|sub-total|net\s+amount|base\s+amount)\b",
    re.IGNORECASE,
)

RE_TAX = re.compile(
    r"\b(?:tax|vat|gst|cgst|sgst|igst|sales\s+tax|service\s+tax)\b",
    re.IGNORECASE,
)

RE_DISCOUNT = re.compile(
    r"\b(?:discount|rebate|deduction|promo|coupon|credit)\b",
    re.IGNORECASE,
)

# Metadata filter: dates, phone numbers, invoice IDs that might resemble amounts
RE_METADATA_IGNORE = re.compile(
    r"\b(?:invoice\s*#?|inv\s*#?|date|tel|phone|ph|fax|po\s*#?|order\s*#?|bill\s*#?|account\s*#?|zip|pin)\b",
    re.IGNORECASE,
)

RE_DATE_FORMATS = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)


# =============================================================================
# Helper Utilities: Text Parsing & Number Normalization
# =============================================================================

def parse_monetary_value(raw_val: str) -> float:
    """Sanitizes and converts currency strings into clean floating point numbers.

    Examples:
        'INR 14,000' -> 14000.0
        '$ 1,250.50' -> 1250.50
        '(500.00)'   -> -500.00 (Accounting negative)
    """
    clean = raw_val.strip()
    is_negative = False

    # Check accounting parentheses: (100.00) -> -100.00
    if clean.startswith("(") and clean.endswith(")"):
        is_negative = True
        clean = clean[1:-1].strip()

    # Strip currency symbols and letters
    clean = re.sub(CURRENCY_PATTERN, "", clean, flags=re.IGNORECASE)
    # Strip whitespace, dollar signs, commas
    clean = clean.replace(",", "").replace("$", "").replace("₹", "").replace("€", "").replace("£", "").strip()

    try:
        val = float(clean)
        return -val if is_negative else val
    except ValueError:
        return 0.0


def extract_currency_symbol(line: str) -> Optional[str]:
    """Detects active currency symbol or code in text."""
    match = RE_CURRENCY.search(line)
    return match.group(0).upper() if match else None


def extract_amounts_from_line(line: str) -> List[Tuple[float, str]]:
    """Extracts all monetary amounts present within a single text line.

    Returns:
        List of (float_value, raw_matched_string)
    """
    results: List[Tuple[float, str]] = []

    # Filter out pure date strings first to avoid false positive amounts
    scrubbed = RE_DATE_FORMATS.sub(" ", line)

    # Search for all number candidates
    candidates = re.findall(NUMBER_PATTERN, scrubbed)
    for cand in candidates:
        # Avoid year numbers (e.g. 2024, 2025, 2026) when alone without decimals or commas
        if cand in {"2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028"} and "." not in cand:
            continue
        val = parse_monetary_value(cand)
        if val > 0 or "(" in line:
            results.append((val, cand))

    return results


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> List[str]:
    """Extracts all text lines from a native PDF document.

    Args:
        pdf_path: Path to target PDF file.

    Returns:
        List[str]: Cleaned list of text lines across all pages.
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    lines: List[str] = []

    if PYMUPDF_AVAILABLE:
        doc = pymupdf.open(path)
        for page in doc:
            text = page.get_text()
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
        doc.close()
    else:
        # Fallback to pypdf
        import pypdf
        reader = pypdf.PdfReader(str(path))
        for page in reader.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

    return lines


# =============================================================================
# Core Verification Engine
# =============================================================================

def verify_invoice_arithmetic(
    text_input: Union[Sequence[str], str, Path],
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """Extracts line items and audits invoice arithmetic balance.

    Audits:
        sum(Items) [+ Tax] [- Discount] == Declared Grand Total

    Args:
        text_input: List of text lines, a newline-delimited string, or path to a PDF file.
        tolerance: Allowable floating-point rounding error (default: 0.01).

    Returns:
        Dict[str, Any]: Comprehensive forensic arithmetic verification report.
    """
    # 1. Normalize input to List[str]
    if isinstance(text_input, (str, Path)) and Path(str(text_input)).is_file() and str(text_input).lower().endswith(".pdf"):
        lines = extract_text_from_pdf(text_input)
    elif isinstance(text_input, str):
        lines = [line.strip() for line in text_input.splitlines() if line.strip()]
    elif isinstance(text_input, Sequence):
        lines = [str(line).strip() for line in text_input if str(line).strip()]
    else:
        raise ValueError(f"Unsupported text_input format: {type(text_input)}")

    detected_currency: Optional[str] = None
    grand_total_record: Optional[Dict[str, Any]] = None
    subtotal_record: Optional[Dict[str, Any]] = None
    tax_records: List[Dict[str, Any]] = []
    discount_records: List[Dict[str, Any]] = []
    line_item_records: List[Dict[str, Any]] = []

    # 2. Iterate through lines and classify financial entries
    for line_idx, line in enumerate(lines):
        # Keep track of currency code
        curr = extract_currency_symbol(line)
        if curr and not detected_currency:
            detected_currency = curr

        amounts = extract_amounts_from_line(line)
        if not amounts:
            continue

        # In financial lines, the final monetary value on the line is typically the line amount
        line_val, raw_num = amounts[-1]

        # A. Grand Total / Total Line
        if RE_GRAND_TOTAL.search(line) and not grand_total_record:
            grand_total_record = {
                "line_index": line_idx,
                "amount": line_val,
                "raw_line": line,
            }
            continue

        # B. Subtotal Line
        if RE_SUBTOTAL.search(line) and not subtotal_record:
            subtotal_record = {
                "line_index": line_idx,
                "amount": line_val,
                "raw_line": line,
            }
            continue

        # C. Tax / GST / VAT Line
        if RE_TAX.search(line):
            tax_records.append({
                "line_index": line_idx,
                "description": line,
                "amount": line_val,
            })
            continue

        # D. Discount / Credit Line
        if RE_DISCOUNT.search(line):
            # Discounts subtract from total
            discount_amount = abs(line_val)
            discount_records.append({
                "line_index": line_idx,
                "description": line,
                "amount": discount_amount,
            })
            continue

        # E. Metadata Line Filter (e.g. "Invoice #: 1024", "Date: 2026-10-31")
        if RE_METADATA_IGNORE.search(line):
            continue

        # F. Valid Line Item
        # Clean description by removing raw amount and currency
        desc = re.sub(NUMBER_PATTERN, "", line)
        desc = re.sub(CURRENCY_PATTERN, "", desc, flags=re.IGNORECASE)
        desc = re.sub(r"[:\-_.]{2,}", " ", desc).strip()
        if not desc:
            desc = f"Item #{len(line_item_records) + 1}"

        line_item_records.append({
            "item_id": len(line_item_records) + 1,
            "description": desc,
            "amount": line_val,
            "raw_line": line,
        })

    # 3. Arithmetic Verification Logic
    items_sum = round(sum(item["amount"] for item in line_item_records), 2)
    total_tax = round(sum(t["amount"] for t in tax_records), 2)
    total_discount = round(sum(d["amount"] for d in discount_records), 2)

    declared_grand_total: Optional[float] = None
    expected_grand_total: float = 0.0
    discrepancy: float = 0.0
    tampering_alarm: bool = False
    verdict: str = "INSUFFICIENT_DATA"
    explanation: str = ""

    if grand_total_record is not None:
        declared_grand_total = round(grand_total_record["amount"], 2)

        # Expected calculation: items + tax - discount
        expected_grand_total = round(items_sum + total_tax - total_discount, 2)
        discrepancy = round(abs(declared_grand_total - expected_grand_total), 2)

        if discrepancy > tolerance:
            tampering_alarm = True
            verdict = "ARITHMETIC_TAMPERING_DETECTED"
            curr_str = f" {detected_currency}" if detected_currency else ""
            explanation = (
                f"ARITHMETIC MISMATCH: Sum of line items ({items_sum:.2f}{curr_str})"
                f"{f' + Tax ({total_tax:.2f})' if total_tax else ''}"
                f"{f' - Discount ({total_discount:.2f})' if total_discount else ''} "
                f"evaluates to {expected_grand_total:.2f}{curr_str}, but the declared Grand Total is "
                f"{declared_grand_total:.2f}{curr_str}. Discrepancy: {discrepancy:.2f}{curr_str}."
            )
        else:
            tampering_alarm = False
            verdict = "ARITHMETIC_CONSISTENT"
            curr_str = f" {detected_currency}" if detected_currency else ""
            explanation = (
                f"ARITHMETIC VERIFIED: Sum of {len(line_item_records)} line items ({items_sum:.2f}{curr_str})"
                f"{f' + Tax ({total_tax:.2f})' if total_tax else ''}"
                f"{f' - Discount ({total_discount:.2f})' if total_discount else ''} "
                f"matches declared Grand Total ({declared_grand_total:.2f}{curr_str}) with zero discrepancy."
            )
    else:
        # Fallback if no explicit Grand Total tag was found
        verdict = "NO_GRAND_TOTAL_FOUND"
        explanation = f"Extracted {len(line_item_records)} line items totaling {items_sum:.2f}, but no explicit Grand Total anchor was identified."

    # Return structured forensic dictionary
    return {
        "status": verdict,
        "is_valid": not tampering_alarm and (grand_total_record is not None),
        "tampering_alarm": tampering_alarm,
        "declared_grand_total": declared_grand_total,
        "calculated_expected_total": expected_grand_total,
        "sum_of_line_items": items_sum,
        "total_tax": total_tax,
        "total_discount": total_discount,
        "discrepancy": discrepancy,
        "currency": detected_currency or "INR",
        "line_item_count": len(line_item_records),
        "line_items": line_item_records,
        "grand_total_record": grand_total_record,
        "subtotal_record": subtotal_record,
        "tax_records": tax_records,
        "discount_records": discount_records,
        "explanation": explanation,
    }


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DocGuard Module 9: Deterministic Arithmetic Logic Verifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to an invoice PDF document to extract and audit.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw text string or path to a text file containing invoice lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pdf:
        print(f"Auditing PDF document arithmetic: {args.pdf}")
        report = verify_invoice_arithmetic(args.pdf)
    elif args.text:
        text_path = Path(args.text)
        if text_path.is_file():
            content = text_path.read_text(encoding="utf-8")
        else:
            content = args.text
        report = verify_invoice_arithmetic(content)
    else:
        # Self-test demonstration with sample invoice lines
        print("No input provided. Running self-test on pristine vs tampered invoice schemas...\n")

        print(">>> Test Case 1: Authentic Pristine Invoice")
        pristine_text = (
            "CITY CLINIC MEDICAL INVOICE\n"
            "Consultation Fee:       INR 1,000\n"
            "Prescription Medicines:  INR 3,000\n"
            "Grand Total:            INR 4,000\n"
        )
        report1 = verify_invoice_arithmetic(pristine_text)
        print(f"  Result: {report1['status']} | Alarm: {report1['tampering_alarm']}")
        print(f"  Explanation: {report1['explanation']}\n")

        print(">>> Test Case 2: Spliced Invoice Total (Fraud Attempt)")
        tampered_text = (
            "CITY CLINIC MEDICAL INVOICE\n"
            "Consultation Fee:       INR 1,000\n"
            "Prescription Medicines:  INR 3,000\n"
            "Grand Total:            INR 14,000\n"
        )
        report2 = verify_invoice_arithmetic(tampered_text)
        print(f"  Result: {report2['status']} | Alarm: {report2['tampering_alarm']}")
        print(f"  Explanation: {report2['explanation']}\n")
        return

    print("=" * 70)
    print("DOCGUARD ARITHMETIC VERIFICATION REPORT")
    print("=" * 70)
    print(f"  * Status:           {report['status']}")
    print(f"  * Tampering Alarm:  {report['tampering_alarm']}")
    print(f"  * Declared Total:   {report['currency']} {report['declared_grand_total']}")
    print(f"  * Calculated Total: {report['currency']} {report['calculated_expected_total']}")
    print(f"  * Discrepancy:      {report['currency']} {report['discrepancy']}")
    print(f"  * Line Items:       {report['line_item_count']} item(s)")
    for item in report["line_items"]:
        print(f"      - {item['description']}: {report['currency']} {item['amount']}")
    print("-" * 70)
    print(f"Explanation: {report['explanation']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
