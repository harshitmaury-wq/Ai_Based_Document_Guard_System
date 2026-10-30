"""
========================================================================================
DocGuard: Multimodal Document Forgery & Manipulation Detection Platform
Module 2: Native PDF & Metadata Forensic Engine (pdf_forensics.py)
----------------------------------------------------------------------------------------
Submodule 2.1: Binary Stream Ingestion & Incremental Update Scanner (%%EOF Analysis)
Submodule 2.2: Metadata Integrity & Timestamp Discrepancy Engine
Submodule 2.3: Font Embedding & Text Stream Consistency Inspector
Submodule 2.4: Unified Forensic Scorer, Risk Aggregator & CLI Test Harness
========================================================================================

Forensic Rationale:
-------------------
When an institution (bank, hospital, university, utility provider) generates an
authentic PDF document, the file is compiled in a single linear pass according to
ISO 32000-1 specifications. The document structure ends with a single cross-reference
pointer ('startxref') and an End-Of-File marker: '%%EOF'.

The Incremental Update Mechanism:
  To allow fast editing of large documents without rewriting gigabytes of objects, the
  PDF standard permits 'Incremental Updates'. When a user opens an invoice in Adobe
  Acrobat, Nitro, or Canva, modifies numbers/dates, and saves the file, the software
  appends:
    1. Modified object dictionaries at the end of the file.
    2. A new cross-reference table ('xref' or '/XRef' stream) mapping the revised objects.
    3. A new 'startxref' offset pointer.
    4. A new '%%EOF' trailer marker.

Forensic Rule:
  A genuine, untampered institutional document should have exactly ONE %%EOF marker.
  The presence of 2 or more %%EOF markers provides definitive binary proof that the
  document was subjected to post-issuance revision (often uncertified alteration).

Metadata & Timestamp Forensics:
  Authentic billing engines (SAP, Tally, QuickBooks, Oracle, ReportLab) emit matching
  /CreationDate and /ModDate timestamps within seconds of each other.
  When an attacker edits an invoice using online tools (iLovePDF, Sejda, Smallpdf) or
  desktop graphics tools (Canva, Photoshop, Nitro Pro), those tools leave identifiable
  producer tags and temporal divergence gaps (ModDate days or months after CreationDate).

Font Embedding Forensics:
  Legitimate institutional invoice templates use a strictly standardized typography
  palette (typically 1 or 2 consistent font families, such as Helvetica or Times-Roman).
  When an attacker uses a PDF editor to overwrite an amount (e.g. changing 4,000 to 14,000),
  the editor frequently injects an 'alien' font dictionary (e.g. embedding a standalone
  ArialMT or Calibri subset solely for that modified number).
  Detecting alien font insertions or excessive font fragmentation unmasks selective text tampering.
"""

from __future__ import annotations

import argparse
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pymupdf  # Official PyMuPDF engine
import pypdf    # Pure-Python PDF AST and metadata engine


# Type alias for supported PDF input types
PDFInputSource = Union[str, Path, bytes, io.BytesIO]


# =============================================================================
# Blacklisted & Whitelisted PDF Software Signatures
# =============================================================================

BLACKLISTED_PRODUCERS = [
    "ilovepdf",
    "sejda",
    "smallpdf",
    "pdf24",
    "sodapdf",
    "pdfescape",
    "docfly",
    "pdf candy",
    "canva",
    "photoshop",
    "gimp",
    "illustrator",
    "coreldraw",
    "inkscape",
    "nitro",
    "foxit phantom",
    "master pdf",
    "pdf-xchange",
    "wondershare",
]

WHITELISTED_PRODUCERS = [
    "reportlab",
    "sap",
    "oracle",
    "tally",
    "quickbooks",
    "jasper",
    "crystal reports",
    "wkhtmltopdf",
    "apache fop",
    "tcpdf",
    "mpdf",
]


# =============================================================================
# Structured Result Containers
# =============================================================================

class PDFBinaryInspectionResult:
    """Structured container for binary PDF inspection findings."""

    def __init__(
        self,
        is_pdf: bool,
        eof_count: int,
        eof_offsets: List[int],
        startxref_offsets: List[int],
        has_incremental_updates: bool,
        pdf_version: Optional[str] = None,
        file_size_bytes: int = 0,
        anomalies: Optional[List[str]] = None,
    ):
        self.is_pdf = is_pdf
        self.eof_count = eof_count
        self.eof_offsets = eof_offsets
        self.startxref_offsets = startxref_offsets
        self.has_incremental_updates = has_incremental_updates
        self.pdf_version = pdf_version
        self.file_size_bytes = file_size_bytes
        self.anomalies = anomalies or []

    def to_dict(self) -> Dict[str, Union[bool, int, str, List[str], List[int], None]]:
        return {
            "is_pdf": self.is_pdf,
            "eof_count": self.eof_count,
            "eof_offsets": self.eof_offsets,
            "startxref_offsets": self.startxref_offsets,
            "has_incremental_updates": self.has_incremental_updates,
            "pdf_version": self.pdf_version,
            "file_size_bytes": self.file_size_bytes,
            "anomalies": self.anomalies,
        }

    def __repr__(self) -> str:
        return (
            f"PDFBinaryInspectionResult(is_pdf={self.is_pdf}, "
            f"eof_count={self.eof_count}, "
            f"has_incremental_updates={self.has_incremental_updates}, "
            f"version={self.pdf_version}, "
            f"anomalies={len(self.anomalies)})"
        )


class PDFMetadataAuditResult:
    """Structured container for metadata and timestamp inspection findings."""

    def __init__(
        self,
        has_metadata: bool,
        creation_date: Optional[str],
        mod_date: Optional[str],
        time_delta_seconds: Optional[float],
        producer: str,
        creator: str,
        is_blacklisted_editor: bool,
        blacklisted_tool: Optional[str],
        is_whitelisted_engine: bool,
        anomalies: Optional[List[str]] = None,
    ):
        self.has_metadata = has_metadata
        self.creation_date = creation_date
        self.mod_date = mod_date
        self.time_delta_seconds = time_delta_seconds
        self.producer = producer
        self.creator = creator
        self.is_blacklisted_editor = is_blacklisted_editor
        self.blacklisted_tool = blacklisted_tool
        self.is_whitelisted_engine = is_whitelisted_engine
        self.anomalies = anomalies or []

    def to_dict(self) -> Dict[str, Union[bool, str, float, List[str], None]]:
        return {
            "has_metadata": self.has_metadata,
            "creation_date": self.creation_date,
            "mod_date": self.mod_date,
            "time_delta_seconds": self.time_delta_seconds,
            "producer": self.producer,
            "creator": self.creator,
            "is_blacklisted_editor": self.is_blacklisted_editor,
            "blacklisted_tool": self.blacklisted_tool,
            "is_whitelisted_engine": self.is_whitelisted_engine,
            "anomalies": self.anomalies,
        }

    def __repr__(self) -> str:
        return (
            f"PDFMetadataAuditResult(producer='{self.producer}', "
            f"blacklisted={self.is_blacklisted_editor}, "
            f"time_delta={self.time_delta_seconds}s, "
            f"anomalies={len(self.anomalies)})"
        )


class PDFFontAuditResult:
    """Structured container for font consistency inspection findings."""

    def __init__(
        self,
        total_fonts_count: int,
        unique_font_families: List[str],
        fonts_details: List[Dict[str, Any]],
        has_alien_font: bool,
        has_font_explosion: bool,
        anomalies: Optional[List[str]] = None,
    ):
        self.total_fonts_count = total_fonts_count
        self.unique_font_families = unique_font_families
        self.fonts_details = fonts_details
        self.has_alien_font = has_alien_font
        self.has_font_explosion = has_font_explosion
        self.anomalies = anomalies or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_fonts_count": self.total_fonts_count,
            "unique_font_families": self.unique_font_families,
            "fonts_details": self.fonts_details,
            "has_alien_font": self.has_alien_font,
            "has_font_explosion": self.has_font_explosion,
            "anomalies": self.anomalies,
        }

    def __repr__(self) -> str:
        return (
            f"PDFFontAuditResult(total_fonts={self.total_fonts_count}, "
            f"families={self.unique_font_families}, "
            f"has_alien_font={self.has_alien_font}, "
            f"anomalies={len(self.anomalies)})"
        )


class UnifiedPDFForensicReport(dict):
    """Master container aggregating binary, metadata, and font forensic audits.

    Inherits from dict so that calling code can treat this object either as a
    structured dictionary:
        report["metadata_risk_score"] -> float (0.0 to 100.0%)
        report["anomalies"]           -> List[str]
    or as a typed forensic report object:
        report.risk_score             -> float (0.0 to 1.0)
        report.verdict                -> str ("CLEAN", "SUSPICIOUS", "HIGH_RISK")
        report.structural_anomalies   -> List[str]
    """

    def __init__(
        self,
        is_pdf: bool,
        risk_score: float,
        verdict: str,
        structural_anomalies: List[str],
        binary_summary: Dict[str, Any],
        metadata_summary: Dict[str, Any],
        font_summary: Dict[str, Any],
    ):
        self.is_pdf = is_pdf
        self.risk_score = risk_score
        self.verdict = verdict
        self.structural_anomalies = structural_anomalies
        self.binary_summary = binary_summary
        self.metadata_summary = metadata_summary
        self.font_summary = font_summary

        # Initialize dictionary keys for direct dictionary access
        super().__init__({
            "is_pdf": self.is_pdf,
            "metadata_risk_score": round(self.risk_score * 100, 1),
            "risk_score_normalized": self.risk_score,
            "verdict": self.verdict,
            "anomalies": self.structural_anomalies,
            "structural_anomalies": self.structural_anomalies,
            "binary_inspection": self.binary_summary,
            "metadata_audit": self.metadata_summary,
            "font_audit": self.font_summary,
        })

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    def __repr__(self) -> str:
        return (
            f"UnifiedPDFForensicReport(is_pdf={self.is_pdf}, "
            f"metadata_risk_score={self['metadata_risk_score']}%, "
            f"verdict='{self.verdict}', "
            f"anomalies={len(self.structural_anomalies)})"
        )


# =============================================================================
# Submodule 2.1: Binary Stream Ingestion & Incremental Update Scanner
# =============================================================================

def read_pdf_bytes(pdf_input: PDFInputSource) -> bytes:
    """Extracts raw bytes from file paths, byte arrays, or memory streams."""
    if isinstance(pdf_input, (str, Path)):
        path = Path(pdf_input)
        if not path.exists():
            raise FileNotFoundError(f"PDF document does not exist: {path.resolve()}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path.resolve()}")
        return path.read_bytes()

    elif isinstance(pdf_input, bytes):
        if len(pdf_input) == 0:
            raise ValueError("Provided PDF byte buffer is empty (0 bytes).")
        return pdf_input

    elif isinstance(pdf_input, io.BytesIO):
        pdf_input.seek(0)
        content = pdf_input.read()
        if len(content) == 0:
            raise ValueError("Provided io.BytesIO stream is empty.")
        return content

    else:
        raise ValueError(
            f"Unsupported PDF input source type: {type(pdf_input).__name__}. "
            "Expected str, Path, bytes, or io.BytesIO."
        )


def scan_pdf_incremental_updates(pdf_input: PDFInputSource) -> PDFBinaryInspectionResult:
    """Scans the raw byte stream of a PDF file for incremental updates and binary red flags."""
    raw_bytes = read_pdf_bytes(pdf_input)
    file_size = len(raw_bytes)
    anomalies: List[str] = []

    header_match = re.search(rb"%PDF-(\d+\.\d+)", raw_bytes[:1024])
    if not header_match:
        return PDFBinaryInspectionResult(
            is_pdf=False,
            eof_count=0,
            eof_offsets=[],
            startxref_offsets=[],
            has_incremental_updates=False,
            pdf_version=None,
            file_size_bytes=file_size,
            anomalies=["File lacks standard %PDF- header within the first 1024 bytes (Not a valid PDF)."],
        )

    pdf_version = header_match.group(1).decode("ascii", errors="replace")
    eof_matches = list(re.finditer(rb"%%EOF", raw_bytes))
    eof_offsets = [m.start() for m in eof_matches]
    eof_count = len(eof_offsets)

    startxref_matches = list(re.finditer(rb"startxref\s+(\d+)", raw_bytes))
    startxref_offsets = [int(m.group(1)) for m in startxref_matches]

    has_incremental_updates = False

    if eof_count == 0:
        anomalies.append("CRITICAL: No %%EOF marker found. The PDF file is truncated or corrupted.")
    elif eof_count == 1:
        last_eof_offset = eof_offsets[-1]
        trailing_bytes = file_size - (last_eof_offset + 5)
        if trailing_bytes > 256:
            anomalies.append(
                f"Suspicious trailing data: Found {trailing_bytes} bytes of hidden data after %%EOF marker."
            )
    else:
        has_incremental_updates = True
        anomalies.append(
            f"INCREMENTAL UPDATE DETECTED: Found {eof_count} separate %%EOF markers. "
            f"The document underwent post-signing modification (revision count = {eof_count})."
        )

    if len(startxref_offsets) > 1 and len(startxref_offsets) != eof_count:
        anomalies.append(
            f"Cross-reference mismatch: Found {len(startxref_offsets)} startxref pointers but {eof_count} %%EOF markers."
        )

    return PDFBinaryInspectionResult(
        is_pdf=True,
        eof_count=eof_count,
        eof_offsets=eof_offsets,
        startxref_offsets=startxref_offsets,
        has_incremental_updates=has_incremental_updates,
        pdf_version=pdf_version,
        file_size_bytes=file_size,
        anomalies=anomalies,
    )


# =============================================================================
# Submodule 2.2: Metadata Integrity & Timestamp Discrepancy Engine
# =============================================================================

def parse_pdf_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Parses standard PDF date strings into timezone-aware Python datetime objects."""
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip()

    match = re.search(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", cleaned)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4) or 0)
        minute = int(match.group(5) or 0)
        second = int(match.group(6) or 0)
        try:
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except ValueError:
            return None

    iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})", cleaned)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))
        hour = int(iso_match.group(4))
        minute = int(iso_match.group(5))
        second = int(iso_match.group(6))
        try:
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except ValueError:
            return None

    return None


def audit_pdf_metadata(pdf_input: PDFInputSource) -> PDFMetadataAuditResult:
    """Audits document metadata for temporal discrepancies and blacklisted editors."""
    raw_bytes = read_pdf_bytes(pdf_input)
    anomalies: List[str] = []

    try:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as err:
        return PDFMetadataAuditResult(
            has_metadata=False,
            creation_date=None,
            mod_date=None,
            time_delta_seconds=None,
            producer="",
            creator="",
            is_blacklisted_editor=False,
            blacklisted_tool=None,
            is_whitelisted_engine=False,
            anomalies=[f"Failed to open PDF metadata stream: {err}"],
        )

    meta = doc.metadata or {}
    producer = str(meta.get("producer") or "").strip()
    creator = str(meta.get("creator") or "").strip()
    creation_date_raw = str(meta.get("creationDate") or "").strip()
    mod_date_raw = str(meta.get("modDate") or "").strip()

    # Dual-Engine Parsing: Cross-validate with pypdf pure-Python parser
    try:
        pypdf_reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        if pypdf_reader.metadata:
            p_producer = str(pypdf_reader.metadata.producer or "").strip()
            p_creator = str(pypdf_reader.metadata.creator or "").strip()
            p_cdate = str(pypdf_reader.metadata.creation_date or "").strip()
            p_mdate = str(pypdf_reader.metadata.modification_date or "").strip()
            if not producer and p_producer:
                producer = p_producer
            if not creator and p_creator:
                creator = p_creator
            if not creation_date_raw and p_cdate:
                creation_date_raw = p_cdate
            if not mod_date_raw and p_mdate:
                mod_date_raw = p_mdate
    except Exception:
        pass

    combined_software = f"{producer} {creator}".lower()
    blacklisted_tool: Optional[str] = None
    is_blacklisted = False
    for tool in BLACKLISTED_PRODUCERS:
        if tool in combined_software:
            is_blacklisted = True
            blacklisted_tool = tool
            anomalies.append(
                f"BLACKLISTED AUTHORING SOFTWARE: Document metadata references '{tool}' "
                f"(Producer: '{producer}', Creator: '{creator}'). Indicates consumer manipulation."
            )
            break

    is_whitelisted = False
    if not is_blacklisted:
        for engine in WHITELISTED_PRODUCERS:
            if engine in combined_software:
                is_whitelisted = True
                break

    dt_creation = parse_pdf_datetime(creation_date_raw)
    dt_mod = parse_pdf_datetime(mod_date_raw)

    time_delta: Optional[float] = None
    if dt_creation and dt_mod:
        time_delta = (dt_mod - dt_creation).total_seconds()

        if time_delta < -60.0:
            anomalies.append(
                f"TEMPORAL FORGERY: /ModDate ({dt_mod.strftime('%Y-%m-%d %H:%M:%S')}) is EARLIER "
                f"than /CreationDate ({dt_creation.strftime('%Y-%m-%d %H:%M:%S')}) by {abs(time_delta):.0f} seconds."
            )
        elif time_delta > 300.0:
            days = time_delta / 86400.0
            if days >= 1.0:
                time_str = f"{days:.1f} days"
            else:
                time_str = f"{time_delta / 3600.0:.1f} hours"
            anomalies.append(
                f"POST-CREATION MODIFICATION GAP: Document was modified {time_str} after initial creation "
                f"(Created: {dt_creation.strftime('%Y-%m-%d')}, Modified: {dt_mod.strftime('%Y-%m-%d')})."
            )

    elif dt_mod and not dt_creation:
        anomalies.append("INCOMPLETE METADATA: /ModDate exists but /CreationDate is missing.")

    has_metadata = bool(producer or creator or creation_date_raw or mod_date_raw)
    if not has_metadata:
        anomalies.append("METADATA STRIPPED: Document contains zero /Info metadata attributes.")

    doc.close()

    return PDFMetadataAuditResult(
        has_metadata=has_metadata,
        creation_date=dt_creation.isoformat() if dt_creation else None,
        mod_date=dt_mod.isoformat() if dt_mod else None,
        time_delta_seconds=time_delta,
        producer=producer,
        creator=creator,
        is_blacklisted_editor=is_blacklisted,
        blacklisted_tool=blacklisted_tool,
        is_whitelisted_engine=is_whitelisted,
        anomalies=anomalies,
    )


# =============================================================================
# Submodule 2.3: Font Embedding & Text Stream Consistency Inspector
# =============================================================================

def clean_font_name(font_name: str) -> str:
    """Extracts base font family by stripping subset prefixes and style suffixes."""
    if not font_name:
        return "Unknown"

    cleaned = font_name.strip()
    if "+" in cleaned:
        cleaned = cleaned.split("+", 1)[1]

    cleaned = re.sub(r"[-_](Bold|Italic|Regular|Roman|Light|Medium|Black)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,](Bold|Italic|Regular)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(MT|PSMT|PS|Pro)$", "", cleaned)

    return cleaned.strip() or font_name


def audit_pdf_fonts(pdf_input: PDFInputSource) -> PDFFontAuditResult:
    """Audits embedded font dictionaries across all document pages."""
    raw_bytes = read_pdf_bytes(pdf_input)
    anomalies: List[str] = []

    try:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as err:
        return PDFFontAuditResult(
            total_fonts_count=0,
            unique_font_families=[],
            fonts_details=[],
            has_alien_font=False,
            has_font_explosion=False,
            anomalies=[f"Failed to inspect PDF font stream: {err}"],
        )

    fonts_details: List[Dict[str, Any]] = []
    font_families_seen: List[str] = []
    seen_xrefs = set()

    font_char_counts: Dict[str, int] = {}
    font_span_samples: Dict[str, List[str]] = {}
    total_characters = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        page_fonts = page.get_fonts(full=True)
        for f_tuple in page_fonts:
            xref = f_tuple[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            font_type = f_tuple[2]
            raw_basefont = f_tuple[3]
            clean_family = clean_font_name(raw_basefont)

            is_subset = "+" in raw_basefont
            subset_tag = raw_basefont.split("+")[0] if is_subset else None

            font_info = {
                "xref": xref,
                "raw_name": raw_basefont,
                "family": clean_family,
                "type": font_type,
                "is_subset": is_subset,
                "subset_tag": subset_tag,
                "page": page_idx + 1,
            }
            fonts_details.append(font_info)

            if clean_family not in font_families_seen and clean_family != "Unknown":
                font_families_seen.append(clean_family)

        try:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        raw_font = span.get("font", "")
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue

                        span_family = clean_font_name(raw_font)
                        char_len = len(span_text)
                        total_characters += char_len

                        font_char_counts[span_family] = font_char_counts.get(span_family, 0) + char_len
                        if span_family not in font_span_samples:
                            font_span_samples[span_family] = []
                        if len(font_span_samples[span_family]) < 3:
                            font_span_samples[span_family].append(span_text[:40])
        except Exception:
            pass

    total_fonts = len(fonts_details)
    num_pages = max(1, len(doc))
    doc.close()

    has_alien_font = False
    has_font_explosion = False

    if total_fonts == 0 or total_characters == 0:
        anomalies.append(
            "IMAGE-ONLY DOCUMENT: Zero embedded vector text fonts found. "
            "Document consists entirely of rasterized images (scanned page or flattened screenshot)."
        )

    else:
        if num_pages == 1 and len(font_families_seen) >= 4:
            has_font_explosion = True
            anomalies.append(
                f"TYPOGRAPHICAL INCONSISTENCY (FONT EXPLOSION): Single-page document contains "
                f"{len(font_families_seen)} distinct font families: {font_families_seen}. "
                "Authentic institutional templates rarely mix more than 2-3 font families."
            )

        if len(font_char_counts) >= 2 and total_characters > 0:
            dominant_family, max_chars = max(font_char_counts.items(), key=lambda x: x[1])
            dominant_ratio = max_chars / float(total_characters)

            for fam, count in font_char_counts.items():
                ratio = count / float(total_characters)
                if fam != dominant_family and ratio <= 0.35 and dominant_ratio >= 0.50:
                    has_alien_font = True
                    sample_texts = font_span_samples.get(fam, ["<snippet>"])
                    anomalies.append(
                        f"ALIEN FONT INSERTION: Detected isolated font '{fam}' ({ratio * 100:.1f}% of text, "
                        f"sample: \"{sample_texts[0]}\"), while primary document typography is "
                        f"'{dominant_family}' ({dominant_ratio * 100:.1f}% of text). Indicates selective text modification."
                    )

    return PDFFontAuditResult(
        total_fonts_count=total_fonts,
        unique_font_families=font_families_seen,
        fonts_details=fonts_details,
        has_alien_font=has_alien_font,
        has_font_explosion=has_font_explosion,
        anomalies=anomalies,
    )


# =============================================================================
# Submodule 2.4: Unified Forensic Scorer, Risk Aggregator & CLI Test Harness
# =============================================================================

def audit_pdf_structure(pdf_input: PDFInputSource) -> UnifiedPDFForensicReport:
    """Master forensic auditor evaluating binary structure, metadata, and typography.

    Risk Calibration Weighting Formula:
      Base Score: 0.0
      + 0.40 : Incremental Update detected (%%EOF >= 2)
      + 0.35 : Blacklisted consumer/online editing software (iLovePDF, Canva, etc.)
      + 0.45 : Temporal Forgery (ModDate < CreationDate)
      + 0.20 : Temporal Discrepancy (ModDate > CreationDate by > 300s)
      + 0.25 : Alien Font Insertion (isolated typography on modified field)
      + 0.15 : Font Explosion on single-page document (>= 4 families)
      + 0.15 : Suspicious trailing payload past final %%EOF marker
      Cap at: 1.0 (100%)

    Verdict Thresholds:
      - Score < 0.25: 'CLEAN' (Pristine, machine-issued document)
      - 0.25 <= Score < 0.60: 'SUSPICIOUS' (Minor anomalies requiring human audit)
      - Score >= 0.60: 'HIGH_RISK' (Definitive binary/metadata evidence of manipulation)

    Args:
        pdf_input: File path, raw bytes, or in-memory stream.

    Returns:
        UnifiedPDFForensicReport: Master risk assessment and aggregated anomalies.
    """
    raw_bytes = read_pdf_bytes(pdf_input)

    # Execute Submodules 2.1, 2.2, and 2.3
    bin_result = scan_pdf_incremental_updates(raw_bytes)

    if not bin_result.is_pdf:
        return UnifiedPDFForensicReport(
            is_pdf=False,
            risk_score=1.0,
            verdict="INVALID_FORMAT",
            structural_anomalies=["Document is not a valid PDF file."],
            binary_summary=bin_result.to_dict(),
            metadata_summary={},
            font_summary={},
        )

    meta_result = audit_pdf_metadata(raw_bytes)
    font_result = audit_pdf_fonts(raw_bytes)

    # -------------------------------------------------------------------------
    # Calibrated Risk Scoring Aggregation
    # -------------------------------------------------------------------------
    score = 0.0
    all_anomalies: List[str] = []

    # 1. Binary Anomalies
    if bin_result.has_incremental_updates:
        score += 0.40
    for a in bin_result.anomalies:
        if a not in all_anomalies:
            all_anomalies.append(f"[BINARY] {a}")
        if "Suspicious trailing data" in a:
            score += 0.15

    # 2. Metadata Anomalies
    if meta_result.is_blacklisted_editor:
        score += 0.35
    if meta_result.time_delta_seconds is not None:
        if meta_result.time_delta_seconds < -60.0:
            score += 0.45  # Temporal forgery attempt
        elif meta_result.time_delta_seconds > 300.0:
            score += 0.20  # Post-creation modification gap

    for a in meta_result.anomalies:
        formatted = f"[METADATA] {a}"
        if formatted not in all_anomalies:
            all_anomalies.append(formatted)

    # 3. Font Anomalies
    if font_result.has_alien_font:
        score += 0.25
    if font_result.has_font_explosion:
        score += 0.15

    for a in font_result.anomalies:
        formatted = f"[FONTS] {a}"
        if formatted not in all_anomalies:
            all_anomalies.append(formatted)

    # If document produced by authentic whitelisted engine with zero anomalies, guarantee 0.0
    if meta_result.is_whitelisted_engine and len(all_anomalies) == 0:
        score = 0.0

    # Cap risk score to [0.0, 1.0]
    final_score = round(min(1.0, max(0.0, score)), 2)

    # Assign Categorical Verdict
    if final_score < 0.25:
        verdict = "CLEAN"
    elif final_score < 0.60:
        verdict = "SUSPICIOUS"
    else:
        verdict = "HIGH_RISK"

    return UnifiedPDFForensicReport(
        is_pdf=True,
        risk_score=final_score,
        verdict=verdict,
        structural_anomalies=all_anomalies,
        binary_summary=bin_result.to_dict(),
        metadata_summary=meta_result.to_dict(),
        font_summary=font_result.to_dict(),
    )


# Backward-compatible alias for master PDF forensic report generator
generate_pdf_forensic_report = audit_pdf_structure


def generate_synthetic_pdf_artifacts(output_dir: str = "dataset/samples") -> Tuple[Path, Path, Path]:
    """Generates synthetic pristine and tampered PDF test documents for demonstration."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n[DocGuard PDF Artifact Generator]")
    print(f"  Generating forensic PDF test documents into: {out_path.resolve()}")

    # 1. Clean Authentic Invoice PDF
    doc_clean = pymupdf.open()
    page_clean = doc_clean.new_page(width=320, height=200)
    page_clean.insert_text((20, 30), "CITY CLINIC MEDICAL INVOICE", fontname="helv", fontsize=12)
    page_clean.insert_text((20, 60), "Consultation Fee:       INR 1,000", fontname="helv", fontsize=10)
    page_clean.insert_text((20, 80), "Prescription Medicines:  INR 3,000", fontname="helv", fontsize=10)
    page_clean.insert_text((20, 110), "Grand Total:            INR 4,000", fontname="helv", fontsize=10)
    doc_clean.set_metadata({
        "producer": "ReportLab PDF Engine v3.5 (Enterprise Accounting)",
        "creator": "Tally Financial ERP",
        "creationDate": "D:20261030100000+00'00'",
        "modDate": "D:20261030100000+00'00'",
    })
    clean_path = out_path / "sample_invoice_pristine.pdf"
    doc_clean.save(clean_path)
    doc_clean.close()

    # 2. Tampered Multi-Revision PDF (Incremental Update)
    raw_clean_bytes = clean_path.read_bytes()
    revision_chunk = [
        b"\n% --- TAMPERED INCREMENTAL UPDATE (ALTERED AMOUNT) ---",
        b"10 0 obj << /Length 38 >> stream\nAltered Grand Total: INR 14,000\nendstream endobj",
        b"xref\n0 1\n0000000000 65535 f \n10 1\n0000000850 00000 n \n",
        b"trailer << /Size 11 /Root 1 0 R >>\nstartxref\n950\n%%EOF\n",
    ]
    multieof_path = out_path / "sample_invoice_tampered_multieof.pdf"
    multieof_path.write_bytes(raw_clean_bytes + b"\n".join(revision_chunk))

    # 3. Tampered Metadata PDF (iLovePDF + Alien Font + Modified Date Gap)
    doc_tamper = pymupdf.open()
    page_tamper = doc_tamper.new_page(width=320, height=200)
    page_tamper.insert_text((20, 30), "CITY CLINIC MEDICAL INVOICE", fontname="helv", fontsize=12)
    page_tamper.insert_text((20, 60), "Consultation Fee:       INR 1,000", fontname="helv", fontsize=10)
    page_tamper.insert_text((20, 80), "Prescription Medicines:  INR 3,000", fontname="helv", fontsize=10)
    # Alien font overwrite on Grand Total: Courier font
    page_tamper.insert_text((20, 110), "Grand Total:            INR 14,000", fontname="cour", fontsize=10)
    doc_tamper.set_metadata({
        "producer": "iLovePDF Online Document Editor",
        "creator": "Mozilla/5.0 Web Canvas",
        "creationDate": "D:20260101100000+00'00'",  # Jan 1, 2026
        "modDate": "D:20260315100000+00'00'",       # Mar 15, 2026 (73 days later)
    })
    meta_tamper_path = out_path / "sample_invoice_tampered_metadata.pdf"
    doc_tamper.save(meta_tamper_path)
    doc_tamper.close()

    print(f"  [+] Pristine PDF Saved:            {clean_path}")
    print(f"  [+] Multi-EOF Tampered PDF Saved:  {multieof_path}")
    print(f"  [+] Metadata Tampered PDF Saved:   {meta_tamper_path}")

    return clean_path, multieof_path, meta_tamper_path


def print_forensic_report_table(report: UnifiedPDFForensicReport, source_name: str = ""):
    """Renders a formatted, colored console report for a PDF forensic audit."""
    print("\n" + "=" * 75)
    print(f"DocGuard PDF Forensic Audit: {source_name or 'Document Stream'}")
    print("=" * 75)
    print(f"  * Document Type:      {'Valid PDF' if report.is_pdf else 'INVALID'}")
    print(f"  * Forensic Verdict:   {report.verdict}")
    print(f"  * Metadata Risk:      {report.risk_score * 100:.0f}% (Score: {report.risk_score:.2f})")
    print(f"  * Revision Count:     {report.binary_summary.get('eof_count', 0)} (%%EOF markers)")
    print(f"  * Producer Tag:       '{report.metadata_summary.get('producer', 'N/A')}'")
    print(f"  * Typography Palette: {report.font_summary.get('unique_font_families', [])}")

    if report.structural_anomalies:
        print("\n  [!] Flagged Structural Anomalies:")
        for idx, anomaly in enumerate(report.structural_anomalies, 1):
            print(f"      {idx}. {anomaly}")
    else:
        print("\n  [+] Zero structural or metadata anomalies detected. Document appears authentic.")
    print("=" * 75)


# =============================================================================
# Standalone Unit Test Suite
# =============================================================================

def run_submodule_2_tests() -> bool:
    """Executes the complete 15-test automated unit test suite for Module 2."""
    print("=" * 75)
    print("DocGuard: Running Verification Test Suite (Module 2: pdf_forensics.py)")
    print("=" * 75)

    test_results = []

    # Helper: Build compliant PDF with custom metadata and fonts
    def build_test_pdf(
        primary_font: str = "helv",
        alien_font: Optional[str] = None,
        producer: str = "ReportLab PDF Engine v3.5",
        creator: str = "Tally Accounting Engine",
        creation_date: str = "D:20261030100000+00'00'",
        mod_date: str = "D:20261030100000+00'00'",
    ) -> bytes:
        doc = pymupdf.open()
        page = doc.new_page(width=300, height=200)
        page.insert_text((20, 30), "CITY CLINIC MEDICAL INVOICE", fontname=primary_font, fontsize=12)
        page.insert_text((20, 60), "Consultation Fee:      INR 1,000", fontname=primary_font, fontsize=10)
        page.insert_text((20, 80), "Prescription Medicine: INR 3,000", fontname=primary_font, fontsize=10)
        if alien_font:
            page.insert_text((20, 110), "Grand Total:           INR 14,000", fontname=alien_font, fontsize=10)
        else:
            page.insert_text((20, 110), "Grand Total:           INR 4,000", fontname=primary_font, fontsize=10)
        doc.set_metadata({
            "producer": producer,
            "creator": creator,
            "creationDate": creation_date,
            "modDate": mod_date,
        })
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    clean_bytes = build_test_pdf()

    # Test 1: Pristine PDF Scan
    bin_res1 = scan_pdf_incremental_updates(clean_bytes)
    assert bin_res1.is_pdf is True and bin_res1.eof_count == 1
    test_results.append(("Pristine Single-Revision PDF Scan", "PASSED"))
    print("  [+] Test 1 Passed: Pristine PDF verified with exactly 1 %%EOF marker.")

    # Test 2: Incremental Update Detection
    bin_res2 = scan_pdf_incremental_updates(clean_bytes + b"\n%%EOF")
    assert bin_res2.eof_count >= 2 and bin_res2.has_incremental_updates is True
    test_results.append(("Incremental Update Tampering Detection", "PASSED"))
    print("  [+] Test 2 Passed: Incremental update detected.")

    # Test 3: Non-PDF Rejection Guard
    bin_res3 = scan_pdf_incremental_updates(b"Not a PDF file content")
    assert bin_res3.is_pdf is False
    test_results.append(("Non-PDF Rejection Guard", "PASSED"))
    print("  [+] Test 3 Passed: Non-PDF content correctly rejected.")

    # Test 4: Authentic Metadata Audit
    meta_res1 = audit_pdf_metadata(clean_bytes)
    assert meta_res1.is_blacklisted_editor is False and meta_res1.time_delta_seconds == 0.0
    test_results.append(("Authentic Metadata Audit (Whitelisted)", "PASSED"))
    print("  [+] Test 4 Passed: Authentic metadata audited.")

    # Test 5: Blacklisted Tool Detection
    ilovepdf_bytes = build_test_pdf(producer="iLovePDF Online Document Editor")
    meta_res2 = audit_pdf_metadata(ilovepdf_bytes)
    assert meta_res2.is_blacklisted_editor is True and meta_res2.blacklisted_tool == "ilovepdf"
    test_results.append(("Blacklisted Editor Detection (iLovePDF)", "PASSED"))
    print("  [+] Test 5 Passed: Blacklisted tool detected ('ilovepdf').")

    # Test 6: Temporal Modification Gap
    tampered_dates_bytes = build_test_pdf(
        creation_date="D:20260101100000+00'00'",
        mod_date="D:20260215100000+00'00'",
    )
    meta_res3 = audit_pdf_metadata(tampered_dates_bytes)
    assert meta_res3.time_delta_seconds > 86400.0 * 40.0
    test_results.append(("Temporal Modification Gap Detection", "PASSED"))
    print("  [+] Test 6 Passed: Temporal discrepancy flagged.")

    # Test 7: Temporal Forgery Detection
    impossible_date_bytes = build_test_pdf(
        creation_date="D:20261030100000+00'00'",
        mod_date="D:20261025100000+00'00'",
    )
    meta_res4 = audit_pdf_metadata(impossible_date_bytes)
    assert any("TEMPORAL FORGERY" in a for a in meta_res4.anomalies)
    test_results.append(("Temporal Forgery Detection (Mod < Create)", "PASSED"))
    print("  [+] Test 7 Passed: Impossible timestamp order flagged.")

    # Test 8: Consistent Typography
    font_res1 = audit_pdf_fonts(clean_bytes)
    assert font_res1.has_alien_font is False and font_res1.has_font_explosion is False
    test_results.append(("Consistent Typography Audit", "PASSED"))
    print("  [+] Test 8 Passed: Consistent font palette verified.")

    # Test 9: Alien Font Injection Detection
    alien_bytes = build_test_pdf(primary_font="helv", alien_font="cour")
    font_res2 = audit_pdf_fonts(alien_bytes)
    assert font_res2.has_alien_font is True
    test_results.append(("Alien Font Injection Detection", "PASSED"))
    print("  [+] Test 9 Passed: Alien font detected.")

    # Test 10: Font Explosion Anomaly
    doc_expl = pymupdf.open()
    page_expl = doc_expl.new_page(width=300, height=300)
    page_expl.insert_text((20, 30), "A", fontname="helv", fontsize=10)
    page_expl.insert_text((20, 60), "B", fontname="tiro", fontsize=10)
    page_expl.insert_text((20, 90), "C", fontname="cour", fontsize=10)
    page_expl.insert_text((20, 120), "D", fontname="symb", fontsize=10)
    expl_bytes = doc_expl.tobytes()
    doc_expl.close()
    font_res3 = audit_pdf_fonts(expl_bytes)
    assert font_res3.has_font_explosion is True
    test_results.append(("Font Explosion Anomaly Detection", "PASSED"))
    print("  [+] Test 10 Passed: Font explosion detected.")

    # Test 11: Image-Only Document Recognition
    doc_img = pymupdf.open()
    doc_img.new_page(width=300, height=200)
    img_bytes = doc_img.tobytes()
    doc_img.close()
    font_res4 = audit_pdf_fonts(img_bytes)
    assert font_res4.total_fonts_count == 0
    test_results.append(("Image-Only Document Recognition", "PASSED"))
    print("  [+] Test 11 Passed: Image-only scanned document recognized.")

    # Test 12: Submodule 2.4: Unified Audit on Pristine Document
    unified_clean = audit_pdf_structure(clean_bytes)
    assert unified_clean.is_pdf is True
    assert unified_clean.risk_score == 0.0
    assert unified_clean.verdict == "CLEAN"
    assert len(unified_clean.structural_anomalies) == 0
    test_results.append(("Unified Audit: Pristine PDF (Score 0.0)", "PASSED"))
    print(f"  [+] Test 12 Passed: Unified audit on clean PDF (Verdict: {unified_clean.verdict}, Score: {unified_clean.risk_score}).")

    # Test 13: Submodule 2.4: Unified Audit on Incremental Update PDF
    multi_eof_doc = clean_bytes + b"\n%%EOF"
    unified_multieof = audit_pdf_structure(multi_eof_doc)
    assert unified_multieof.risk_score >= 0.40
    assert unified_multieof.verdict in ("SUSPICIOUS", "HIGH_RISK")
    assert any("INCREMENTAL UPDATE DETECTED" in a for a in unified_multieof.structural_anomalies)
    test_results.append(("Unified Audit: Incremental Update (>= 0.40)", "PASSED"))
    print(f"  [+] Test 13 Passed: Incremental update flagged in unified report (Score: {unified_multieof.risk_score}).")

    # Test 14: Submodule 2.4: Unified Audit on Heavily Tampered Document
    heavy_tamper_bytes = build_test_pdf(
        primary_font="helv",
        alien_font="cour",
        producer="iLovePDF Online Document Editor",
        creation_date="D:20260101100000+00'00'",
        mod_date="D:20260315100000+00'00'",
    ) + b"\n%%EOF"
    unified_heavy = audit_pdf_structure(heavy_tamper_bytes)
    assert unified_heavy.risk_score >= 0.85
    assert unified_heavy.verdict == "HIGH_RISK"
    assert len(unified_heavy.structural_anomalies) >= 3
    test_results.append(("Unified Audit: Heavily Tampered (HIGH_RISK)", "PASSED"))
    print(f"  [+] Test 14 Passed: Heavily tampered PDF flagged as HIGH_RISK (Score: {unified_heavy.risk_score * 100:.0f}%).")

    # Test 15: Submodule 2.4: Invalid Document Rejection
    unified_invalid = audit_pdf_structure(b"Plain text non-pdf data")
    assert unified_invalid.is_pdf is False
    assert unified_invalid.verdict == "INVALID_FORMAT"
    test_results.append(("Unified Audit: Non-PDF Rejection", "PASSED"))
    print("  [+] Test 15 Passed: Non-PDF rejected with INVALID_FORMAT verdict.")

    print("=" * 75)
    print("Module 2 (pdf_forensics.py) Full Verification Summary:")
    for test_name, status in test_results:
        print(f"  * {test_name:<44}: {status}")
    print("=" * 75)
    print("SUCCESS: All 15 unit tests passed! Module 2 is 100% complete and verified.")
    return True


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DocGuard Module 2: Native PDF & Metadata Forensic Engine (pdf_forensics.py)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Execute the 15-test automated unit test suite."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Generate synthetic PDF test files and print forensic audit reports."
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Inspect any custom PDF document file from disk."
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset/samples", help="Directory to save generated test artifacts."
    )

    args = parser.parse_args()
    run_all = not (args.test or args.demo or args.input)

    if args.test or run_all:
        run_submodule_2_tests()

    if args.demo or run_all:
        clean_p, multieof_p, meta_p = generate_synthetic_pdf_artifacts(output_dir=args.output_dir)

        # Run unified audits and print reports
        rep_clean = audit_pdf_structure(clean_p)
        print_forensic_report_table(rep_clean, "1. Authentic Medical Invoice (Pristine)")

        rep_multieof = audit_pdf_structure(multieof_p)
        print_forensic_report_table(rep_multieof, "2. Post-Signing Incremental Update (Multi-%%EOF)")

        rep_meta = audit_pdf_structure(meta_p)
        print_forensic_report_table(rep_meta, "3. Forged Metadata (iLovePDF + Alien Font + Date Gap)")

    if args.input:
        input_path = Path(args.input)
        report = audit_pdf_structure(input_path)
        print_forensic_report_table(report, str(input_path.name))
