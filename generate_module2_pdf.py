#!/usr/bin/env python3
"""
DocGuard Module 2 Explanation PDF Generator
Compiles a publication-quality pedagogical report from MODULE_2_EXPLANATION.md
into MODULE_2_EXPLANATION.pdf using ReportLab.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y' on every page,
    along with running header rules.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 760, "DOCGUARD FORENSIC ARCHITECTURE | MODULE 2: NATIVE PDF & METADATA FORENSICS")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(40, 752, 572, 752)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(40, 42, 572, 42)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(40, 30, "DocGuard Digital Forensics Platform | Confidential & Proprietary")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_str)
        self.restoreState()


def create_module2_pdf(output_pdf_path: str = "MODULE_2_EXPLANATION.pdf"):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom typography hierarchy
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1e40af'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'Heading3_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#0f172a')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1e293b')
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title block
    story.append(Paragraph("DocGuard — Module 2 Pedagogical & Technical Guide", title_style))
    story.append(Paragraph("Understanding the Native PDF & Metadata Forensic Engine (pdf_forensics.py)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceBefore=0, spaceAfter=12))

    # Section 1
    story.append(Paragraph("1. What is Module 2 Doing? (The Objective in Simple Language)", h1_style))
    p1 = (
        "Imagine a fraudster wants to tamper with an electronic medical bill or bank statement in PDF format. "
        "Instead of printing it out, scanning it, or using complex photo manipulation software, they open the PDF in an "
        "online editor like <b>iLovePDF</b> or <b>Canva</b>, click on the total amount <i>INR 4,000</i>, type <i>INR 14,000</i>, "
        "and hit <b>Save</b>.<br/><br/>"
        "To human eyes reading the document on a screen, the PDF looks completely crisp, sharp, and authentic. "
        "There are no pixel blurs, no low-resolution artifacts, and no obvious visual flaws. "
        "Because an investigator does not possess the issuer's original master copy, visual inspection alone fails.<br/><br/>"
        "<b>Module 2 is the low-level forensic investigator that inspects the file's internal DNA and digital paper trail.</b>"
    )
    story.append(Paragraph(p1, body_style))

    # Callout Box: Core Mission
    callout_data = [[
        Paragraph(
            "<b>The Core Forensic Mission:</b><br/>"
            "Before a single pixel is ever rendered or passed to deep neural networks, Module 2 executes a deep structural autopsy "
            "directly on the carrier file bytes:<br/>"
            "• <b>Physical File Structure (Binary Autopsy):</b> Uncovers post-issuance saving sessions, hidden revisions, and unauthorized trailing payloads.<br/>"
            "• <b>Timestamp & Authoring Audit (Metadata Forensics):</b> Decodes ASN.1 internal dates, audits modification gaps, and checks software blacklists.<br/>"
            "• <b>Typographical Uniformity (Font DNA):</b> Extracts embedded font dictionaries to detect isolated 'Alien Font' text alterations.<br/>"
            "• <b>Calibrated Risk Scoring Brain:</b> Fuses independent physical indicators into an empirical risk score (0.0 Clean to 1.0 Tampered).",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[532])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('LINELEFT', (0, 0), (0, -1), 3.5, colors.HexColor('#2563eb')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # Section 2
    story.append(Paragraph("2. How Does Module 2 Achieve This? (The 4-Stage Forensic Pipeline)", h1_style))
    story.append(Paragraph(
        "Module 2 processes PDF documents through a sequential multi-stage inspection architecture:",
        body_style
    ))

    pipeline_items = [
        ("Stage 1: Binary File Autopsy",
         "Ingests the raw byte stream in memory. Validates the ISO 32000-1 magic header (%PDF-), scans backwards for %%EOF "
         "markers to detect non-destructive incremental updates, verifies cross-reference tables (startxref), and checks for hidden post-EOF payloads."),
        ("Stage 2: Metadata & Temporal Audit",
         "Extracts internal /Info and XMP metadata dictionaries. Decodes ASN.1 timestamps (D:YYYYMMDD...) into UTC datetimes, "
         "computes the creation-to-modification delta (Δt), flags negative time travel (date spoofing), and screens authoring signatures against blacklists."),
        ("Stage 3: Typography & Font DNA Audit",
         "Inspects low-level font descriptors and character distributions via page.get_text('dict'). Measures character ratios "
         "per font family to pinpoint isolated 'Alien Font Injections' (e.g. Courier total inside a Helvetica invoice) and flags typographic font explosions."),
        ("Stage 4: Calibrated Multi-Vector Risk Scoring",
         "Fuses binary, temporal, authoring, and typographic anomalies into an empirical risk score S_risk in [0.0, 1.0]. "
         "Assigns categorical verdicts (CLEAN, SUSPICIOUS, HIGH_RISK) accompanied by deterministic, court-grade explanations.")
    ]

    for stage_title, stage_desc in pipeline_items:
        story.append(Paragraph(f"<b>• {stage_title}:</b> {stage_desc}", body_style))

    story.append(Spacer(1, 10))

    # Section 3
    story.append(Paragraph("3. Comprehensive Glossary of Technical, Forensic & ML Concepts", h1_style))
    story.append(Paragraph(
        "Below is an exhaustive explanation of every technical keyword, binary file principle, forensic indicator, "
        "and machine learning concept implemented in Module 2.",
        body_style
    ))

    # Group A
    story.append(Paragraph("A. PDF Binary Architecture & ISO 32000-1 Standards", h2_style))

    story.append(Paragraph("1. ISO 32000-1 Specification", h3_style))
    story.append(Paragraph(
        "The international standard defining the Portable Document Format (PDF). It governs how objects (text, fonts, "
        "vector shapes, bitmap images) are stored, compressed, indexed, and cross-referenced. Deviations from this standard "
        "provide deterministic physical evidence of third-party tampering.",
        body_style
    ))

    story.append(Paragraph("2. Magic Byte Header (%PDF-)", h3_style))
    story.append(Paragraph(
        "The mandatory leading sequence (e.g. '%PDF-1.4') required within the first 1024 bytes of an authentic PDF. "
        "DocGuard verifies this header to prevent arbitrary file upload attacks, extension spoofing, or feeding corrupted binaries.",
        body_style
    ))

    story.append(Paragraph("3. Incremental Updates & Multiple %%EOF Markers", h3_style))
    story.append(Paragraph(
        "The PDF format allows appending changes to the end of an existing file rather than re-writing the entire document. "
        "When an attacker opens a PDF in an editor and re-saves it, the editor appends modified objects, a new cross-reference "
        "table, and a second <b>%%EOF (End Of File)</b> marker.<br/>"
        "• <b>Forensic Signature:</b> An authentic, automated institutionally-issued invoice has <b>exactly 1 %%EOF marker</b>. "
        "A document containing 2 or more %%EOF markers physically proves that the document underwent multiple post-issuance saving sessions.",
        body_style
    ))

    story.append(Paragraph("4. Cross-Reference Tables (xref) and startxref Pointers", h3_style))
    story.append(Paragraph(
        "PDFs are random-access databases of numbered objects. The cross-reference table (xref) indexes the exact byte offset "
        "of each object, and 'startxref' tells the reader where the index begins. In multi-session edits, each revision appends "
        "a new xref block and startxref pointer.",
        body_style
    ))

    story.append(Paragraph("5. Trailing Hidden Payloads (Post-EOF Injections)", h3_style))
    story.append(Paragraph(
        "Standard PDF readers terminate parsing upon reaching the final %%EOF delimiter. Any bytes appended past this marker "
        "are completely hidden from human viewers. Forgers and attackers exploit this to conceal secondary documents, keys, "
        "or malicious scripts. DocGuard scans beyond the terminal %%EOF to expose hidden payloads.",
        body_style
    ))

    # Group B
    story.append(Paragraph("B. Metadata & Temporal Forensic Analysis", h2_style))

    story.append(Paragraph("1. Internal /Info Dictionary vs. File System Timestamps", h3_style))
    story.append(Paragraph(
        "Operating system timestamps (e.g. Windows 'Date Modified') change whenever a file is copied, emailed, or downloaded. "
        "In contrast, the PDF internal /Info catalog travels inside the document bytes and contains /CreationDate, /ModDate, "
        "/Producer, and /Creator, preserving the authentic digital paper trail.",
        body_style
    ))

    story.append(Paragraph("2. ASN.1 Standardized Timestamp Format", h3_style))
    story.append(Paragraph(
        "PDF dates follow the ASN.1 format: <code>D:YYYYMMDDHHmmSS[+|-]HH'mm'</code>. "
        "For example, <code>D:20261030143000+05'30'</code> represents October 30, 2026 at 14:30:00 UTC+5:30. "
        "DocGuard decodes these strings into exact Python datetime instances with full timezone awareness.",
        body_style
    ))

    story.append(Paragraph("3. Temporal Modification Gap (Δt)", h3_style))
    story.append(Paragraph(
        "The mathematical difference between the modification timestamp and the creation timestamp: "
        "<code>Δt = ModDate - CreationDate</code>.<br/>"
        "In automated accounting, an invoice is generated and finalized instantaneously (Δt < 60s). "
        "A gap where Δt exceeds 86,400 seconds (24 hours) demonstrates that the document was modified in a separate session days or months later.",
        body_style
    ))

    story.append(Paragraph("4. Impossible Temporal Inversion (Negative Time Travel)", h3_style))
    story.append(Paragraph(
        "A condition where ModDate < CreationDate (Δt < -60s). Because an authentic file cannot be modified prior to its creation, "
        "a negative time gap is mathematical proof of metadata forgery, indicating that a fraudster forged dates manually but failed "
        "to adjust both fields synchronously.",
        body_style
    ))

    story.append(Paragraph("5. Software Fingerprinting: Whitelisting vs. Blacklisting", h3_style))
    story.append(Paragraph(
        "• <b>Enterprise Whitelist:</b> Automated, unalterable billing engines such as <i>ReportLab, SAP ERP, Tally, Oracle NetSuite, QuickBooks</i>.<br/>"
        "• <b>Consumer Blacklist:</b> Interactive editors designed for manual document alteration: <i>iLovePDF, Sejda, Canva, Adobe Photoshop, Nitro Pro, Smallpdf</i>.<br/>"
        "An institutional hospital invoice whose internal Producer reads 'iLovePDF' is an immediate, definitive giveaway of human tampering.",
        body_style
    ))

    # Group C
    story.append(Paragraph("C. Typography & Font DNA Forensics", h2_style))

    story.append(Paragraph("1. Embedded Font Dictionaries", h3_style))
    story.append(Paragraph(
        "PDF text objects reference font dictionaries that define character glyph shapes, widths, and encodings. "
        "Authentic corporate documents use consistent typography defined in institutional document templates.",
        body_style
    ))

    story.append(Paragraph("2. The 'Alien Font Injection' Attack Signature", h3_style))
    story.append(Paragraph(
        "When an attacker opens a PDF in an editor to alter a specific number, the editor often lacks the exact font "
        "embedded by the original billing software. The editor creates a brand-new font dictionary or injects a fallback "
        "system font (e.g. Courier or Arial) solely for that modified text box.<br/>"
        "• <b>Forensic Signal:</b> DocGuard audits text span character counts. If a document is 75% Helvetica but has an isolated "
        "25% block of Courier containing 'Grand Total: INR 14,000', DocGuard flags an Alien Font Insertion and extracts the exact altered text.",
        body_style
    ))

    story.append(Paragraph("3. Typographical Font Explosion", h3_style))
    story.append(Paragraph(
        "A single-page document featuring 4 or more distinct font families. Professional billing templates adhere to strict "
        "typographic guidelines (typically 1 header font and 1 body font). Font explosions indicate composite document splicing.",
        body_style
    ))

    story.append(Paragraph("4. Scanned / Flattened Document Recognition", h3_style))
    story.append(Paragraph(
        "Identifies PDFs containing zero vector fonts and zero searchable text—documents that consist solely of a raster image. "
        "This alerts downstream modules to rely primarily on visual and pixel-level forensic streams (Module 1 ELA and CNN backbones).",
        body_style
    ))

    # Group D
    story.append(Paragraph("D. Mathematical Fusion & Machine Learning Connections", h2_style))

    story.append(Paragraph("1. Multi-Vector Evidence Fusion", h3_style))
    story.append(Paragraph(
        "No single forensic signal is completely immune to false positives. However, when independent physical planes "
        "(Binary, Metadata, Typography) compound—such as an incremental update + a blacklisted tool + an alien font—the "
        "joint probability of innocent coincidence drops to near-zero.",
        body_style
    ))

    story.append(Paragraph("2. Calibrated Empirical Risk Formulation", h3_style))
    story.append(Paragraph(
        "The risk engine computes an aggregated risk score: "
        "<code>S_risk = min(1.0, Σ w_i · 1_{A_i})</code> where w_i represents empirical severity weights:",
        body_style
    ))

    # Risk Weight Table
    weight_data = [
        [Paragraph("Forensic Anomaly Indicator (A_i)", table_header),
         Paragraph("Weight (w_i)", table_header),
         Paragraph("Forensic Justification & Physical Impact", table_header)],
        [Paragraph("Incremental Update (N_EOF >= 2)", table_cell_bold),
         Paragraph("0.40", table_cell),
         Paragraph("Physical proof of a secondary post-creation save session.", table_cell)],
        [Paragraph("Blacklisted Consumer Software", table_cell_bold),
         Paragraph("0.35", table_cell),
         Paragraph("Document passed through an interactive editor (e.g. iLovePDF).", table_cell)],
        [Paragraph("Alien Font Injection", table_cell_bold),
         Paragraph("0.30", table_cell),
         Paragraph("Isolated font introduced specifically for altered character sequences.", table_cell)],
        [Paragraph("Temporal Forgery (Mod < Create)", table_cell_bold),
         Paragraph("0.35", table_cell),
         Paragraph("Impossible timestamp order (conclusive date manipulation).", table_cell)],
        [Paragraph("Post-Creation Gap (Δt > 24h)", table_cell_bold),
         Paragraph("0.25", table_cell),
         Paragraph("Document re-saved long after the original generation event.", table_cell)],
        [Paragraph("Hidden Trailing Data (> 32 bytes)", table_cell_bold),
         Paragraph("0.25", table_cell),
         Paragraph("Unreferenced bytes or payloads appended past final %%EOF.", table_cell)],
        [Paragraph("Font Explosion (>= 4 fonts)", table_cell_bold),
         Paragraph("0.20", table_cell),
         Paragraph("Typographical inconsistency indicating spliced layout elements.", table_cell)],
        [Paragraph("Stripped / Missing Metadata", table_cell_bold),
         Paragraph("0.15", table_cell),
         Paragraph("Complete absence of standard /Info metadata headers.", table_cell)],
    ]
    weight_table = Table(weight_data, colWidths=[160, 65, 307])
    weight_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(weight_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Connection to Bayesian Priors in Multimodal Learning", h3_style))
    story.append(Paragraph(
        "In multimodal machine learning, the Metadata Risk Score functions as an <b>informative Bayesian prior</b>. "
        "When Stream 3 (Decision Fusion Brain) evaluates a document with S_risk = 0.80, the deep vision networks do not begin "
        "from an uninformative 50/50 prior; they are mathematically informed that the carrier file is structurally compromised. "
        "Conversely, if S_risk = 0.00, the vision network requires significantly stronger pixel-level visual evidence before "
        "triggering a false alarm.",
        body_style
    ))

    story.append(Paragraph("4. Explainable AI (XAI) & Judicial Defense", h3_style))
    story.append(Paragraph(
        "Deep convolutional networks operate as 'black boxes' that output a single probability score without reasoning. "
        "In financial fraud investigations and courtrooms, unexplainable predictions are legally inadmissible. "
        "Module 2 delivers <b>100% explainable, deterministic evidence</b>: every point added to the risk score is accompanied "
        "by exact byte offsets, timestamp discrepancies, and extracted text strings.",
        body_style
    ))

    # Summary Table
    story.append(Spacer(1, 6))
    story.append(Paragraph("Summary: How Module 2 Signals Protect the System", h2_style))

    summary_data = [
        [Paragraph("Extracted Signal", table_header),
         Paragraph("Layer Inspected", table_header),
         Paragraph("What It Detects", table_header),
         Paragraph("Downstream Consumer", table_header)],
        [Paragraph("%%EOF Marker Count", table_cell_bold),
         Paragraph("Raw Byte Stream", table_cell),
         Paragraph("Post-signing modifications, multi-session saves", table_cell),
         Paragraph("Risk Engine & Audit Log", table_cell)],
        [Paragraph("Trailing Byte Count", table_cell_bold),
         Paragraph("Post-EOF Bytes", table_cell),
         Paragraph("Hidden payloads, appended files, malware", table_cell),
         Paragraph("Security Alert", table_cell)],
        [Paragraph("ASN.1 Timestamp Delta (Δt)", table_cell_bold),
         Paragraph("/Info Dictionary", table_cell),
         Paragraph("Human editing sessions, impossible dates", table_cell),
         Paragraph("Risk Engine & Timeline", table_cell)],
        [Paragraph("Software Producer Tag", table_cell_bold),
         Paragraph("/Producer & /Creator", table_cell),
         Paragraph("Consumer editing tools (iLovePDF, Canva)", table_cell),
         Paragraph("Risk Engine & Tool Alert", table_cell)],
        [Paragraph("Font Span Character Ratios", table_cell_bold),
         Paragraph("Text Span Dictionaries", table_cell),
         Paragraph("Alien font injections, altered numbers", table_cell),
         Paragraph("Risk Engine & Text Localizer", table_cell)],
        [Paragraph("Calibrated Risk Score (S_risk)", table_cell_bold),
         Paragraph("Unified Aggregator", table_cell),
         Paragraph("Holistic structural risk index (0.0 to 1.0)", table_cell),
         Paragraph("Decision Fusion Brain", table_cell)],
    ]

    summary_table = Table(summary_data, colWidths=[120, 100, 192, 120])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(summary_table)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"SUCCESS: Generated {output_pdf_path}")


if __name__ == "__main__":
    create_module2_pdf()
