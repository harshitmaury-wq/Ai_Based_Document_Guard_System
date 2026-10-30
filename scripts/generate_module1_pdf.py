"""
DocGuard: Module 1 Technical Guide PDF Generator
Uses ReportLab to compile a publication-grade, styled PDF of MODULE_1_EXPLANATION.md
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Custom canvas that computes total pages dynamically for a 'Page X of Y' footer."""
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

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "DocGuard — Module 1 Technical & Pedagogical Guide")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "In-Memory Forensic Preprocessor")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Running Footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.drawString(54, 36, "Confidential — DocGuard Digital Document Forensics Platform")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)

        self.restoreState()


def create_module1_pdf(output_pdf_path: str = "MODULE_1_EXPLANATION.pdf"):
    pdf_path = Path(output_pdf_path)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    callout_style = ParagraphStyle(
        "Callout_Text",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("DocGuard: Forensic Preprocessor", title_style))
    story.append(Paragraph("Module 1 Technical & Pedagogical Guide — Understanding the Signal Extraction Engine", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=14))

    # Section 1
    story.append(Paragraph("1. What is Module 1 Doing? (The Objective in Simple Language)", h1_style))
    story.append(Paragraph(
        "Imagine someone hands you a printed medical receipt where the total amount has been altered from "
        "<b>INR 4,000</b> to <b>INR 14,000</b> using Photoshop or an AI editing tool. To the human eye, the numbers look "
        "completely normal. Standard Optical Character Recognition (OCR) simply reads 'INR 14,000'. And because an auditor "
        "rarely has the clinic's original reference document, standard hash checks or visual inspections fail.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Module 1 is the digital forensic investigator that examines the physical physics underneath the surface pixels.</b>",
        body_style,
    ))
    story.append(Paragraph(
        "Its objective is <b>Forensic Signal Extraction</b>: before passing any document to deep neural networks, "
        "Module 1 extracts two hidden physical signatures that humans cannot see, but which prove whether an image has been altered:",
        body_style,
    ))

    # Bullet points
    p1 = "<b>1. Compression Equilibrium Discrepancies (ELA):</b> Inspects whether different areas of the document have been compressed at different times (exposing altered numbers, modified dates, or inserted text)."
    p2 = "<b>2. Paper Grain & Noise Discontinuities:</b> Analyzes the microscopic physical paper texture and sensor noise across the page to expose unnaturally smooth ('sterile') patches left by AI inpainting or digital erasing."
    story.append(Paragraph(f"• {p1}", body_style))
    story.append(Paragraph(f"• {p2}", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "All calculations execute <b>100% in system memory (RAM)</b> via virtual streams (io.BytesIO), eliminating disk I/O latency "
        "and preventing Windows file-locking crashes.",
        body_style,
    ))

    # Section 2
    story.append(Spacer(1, 8))
    story.append(Paragraph("2. How Does Module 1 Achieve This? (Three-Stage Workflow)", h1_style))
    
    stages_data = [
        [
            Paragraph("<b>Stage</b>", body_bold),
            Paragraph("<b>Core Operation</b>", body_bold),
            Paragraph("<b>Forensic Purpose & Physical Output</b>", body_bold),
        ],
        [
            Paragraph("<b>Stage 1:<br/>Standardization</b>", body_style),
            Paragraph("Converts input to RGB, applies EXIF transposition, and composites transparency over pure white paper.", body_style),
            Paragraph("Guarantees clean, upright 3-channel images without black-border transparency artifacts.", body_style),
        ],
        [
            Paragraph("<b>Stage 2:<br/>In-Memory ELA</b>", body_style),
            Paragraph("Re-compresses image in RAM at Quality 95 and computes absolute difference: |I_orig - I_resaved|.", body_style),
            Paragraph("Untouched text remains dark; newly modified numbers glow bright white due to first-time lossy quantization.", body_style),
        ],
        [
            Paragraph("<b>Stage 3:<br/>Noise Residuals</b>", body_style),
            Paragraph("Applies a 3x3 non-linear median filter and subtracts smoothed image: |I - I_smooth|.", body_style),
            Paragraph("Isolates paper fiber grain and sensor noise. Exposes digitally erased/inpainted patches that lack natural texture.", body_style),
        ],
    ]
    t_stages = Table(stages_data, colWidths=[1.3 * inch, 2.7 * inch, 2.8 * inch])
    t_stages.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_stages)

    # Section 3: Technical Concepts
    story.append(Spacer(1, 10))
    story.append(Paragraph("3. Comprehensive Glossary of Technical & ML Concepts", h1_style))
    story.append(Paragraph("Below is an explanation of every technical term and machine learning concept used in Module 1.", body_style))

    # Category A
    story.append(Paragraph("A. Digital Image Fundamentals & Color Spaces", h2_style))
    story.append(Paragraph(
        "<b>• Pixels & RGB Color Space:</b> A digital image is a 2D matrix of colored dots (pixels). Standard images use RGB (Red, Green, Blue), "
        "where each pixel has 3 numbers from 0 to 255. (255, 255, 255) is pure white paper, and (0, 0, 0) is black printed ink.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• The Alpha Channel & 'Black Border' Trap:</b> In PNG scans or digital stamps, a 4th channel (Alpha) controls transparency. "
        "If an image is naively converted to RGB, transparent pixels automatically turn pitch-black (0, 0, 0). "
        "On white paper, this creates an extreme artificial gradient jump [255 -> 0] that tricks neural networks into false alarms. "
        "Module 1 solves this by <b>alpha-compositing onto a white canvas</b>, making transparency appear as authentic white paper.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• EXIF Orientation & Matrix Transposition:</b> Mobile phone cameras record orientation metadata (e.g. 'Rotate 90 deg') "
        "without physically rotating the pixel grid. Module 1 reads this EXIF tag and transposes the matrix upright so text is horizontal.",
        body_style,
    ))

    # Category B
    story.append(Spacer(1, 4))
    story.append(Paragraph("B. Compression Physics & Error Level Analysis (ELA)", h2_style))
    story.append(Paragraph(
        "<b>• Lossy vs. Lossless Compression:</b> PNG and TIFF are lossless (zero data lost). JPEG is lossy—it permanently discards fine "
        "details the human eye cannot easily notice to make files smaller.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Discrete Cosine Transform (DCT):</b> JPEG splits images into 8x8 pixel blocks and converts spatial pixels into 64 frequency waves. "
        "Low frequencies represent paper background; high frequencies represent sharp character edges.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Quantization & Quantization Matrices:</b> The lossy step in JPEG. Frequency coefficients are divided by a Quantization Table "
        "and rounded to whole integers. Fractional remainders are discarded forever.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Quantization Equilibrium:</b> When an unedited JPEG is re-saved at the same quality level, its pixels have already been divided "
        "by that quantization table. Dividing again causes near-zero rounding error (Delta ~ 0). This stability is called Quantization Equilibrium.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• The ELA Tampering Anomaly:</b> When an attacker modifies numbers (e.g. altering INR 4,000 to INR 14,000), those altered digits "
        "are saved for the <i>very first time</i>. When Module 1 compresses the whole document at Quality 95 and computes the absolute difference, "
        "untouched text in equilibrium stays dark, while the newly inserted digits undergo active lossy quantization and <b>glow bright white</b>.",
        body_style,
    ))

    # Category C
    story.append(Spacer(1, 4))
    story.append(Paragraph("C. Spatial Noise, Filtering & AI Inpainting", h2_style))
    story.append(Paragraph(
        "<b>• Sensor Noise & Paper Grain:</b> Authentic scans possess microscopic high-frequency texture caused by camera photon shot noise, "
        "thermal noise, and the physical cellulose grain of authentic wood fibers in paper.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Non-Linear Median Filtering vs. Linear Blur:</b> Standard linear Gaussian blur averages neighboring pixels, turning black text "
        "and white paper into blurry gray halos. A 3x3 median filter sorts neighborhood pixels and picks the middle value. It is <i>edge-preserving</i>: "
        "it strips isolated noise dots while keeping character edges crisp.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Spatial Noise Residual:</b> Subtracting the median-smoothed image from the original isolates the high-pass texture: R(x, y) = |I - I_smooth|.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• The 'Sterile Patch' Anomaly:</b> When an attacker uses Adobe Photoshop 'Content-Aware Fill' or Generative AI (Stable Diffusion) "
        "to erase a stamp or text, the AI algorithm paints over the area smoothly. <b>The AI cannot recreate the microscopic physical noise of the scanner or paper.</b> "
        "In the noise residual map, authentic paper has a healthy grain variance, while the erased area has zero noise (mean = 0.0), exposing the alteration immediately.",
        body_style,
    ))

    # Category D
    story.append(Spacer(1, 4))
    story.append(Paragraph("D. Machine Learning & Deep Learning Terminology", h2_style))
    story.append(Paragraph(
        "<b>• Why Raw RGB Confuses Standard CNNs:</b> If a Convolutional Neural Network (CNN) is trained only on raw RGB images, it often memorizes "
        "document layouts and company logos rather than forensic tampering cues. It may flag a genuine document simply because it has never seen that clinic's logo.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Feature Decoupling & Inductive Bias:</b> By providing pre-computed ELA maps and noise residuals, we decouple the physics of tampering from "
        "the document's visual text. Stream 1 (EfficientNet-B0) is forced to specialize in compression artifacts, while Stream 2 (ResNet-50) looks at font geometry.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Out-of-Distribution (OOD) Robustness:</b> Because Module 1 is based on physical laws (JPEG quantization math and sensor noise variance) "
        "rather than memorized visual templates, it can detect alterations produced by brand-new generative AI tools that did not exist during training.",
        body_style,
    ))

    # Category E
    story.append(Spacer(1, 4))
    story.append(Paragraph("E. In-Memory Streaming & Operating System Architecture", h2_style))
    story.append(Paragraph(
        "<b>• Virtual RAM Streams (io.BytesIO):</b> Standard programs write temporary images to the hard drive (C:\\temp\\image.jpg). "
        "io.BytesIO creates a virtual file in RAM. Python interacts with it like a real file, but it runs at sub-millisecond RAM speeds.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>• Windows File Locking (WinError 32):</b> On Windows, opening a disk file locks it. If another thread or antivirus tries to access it, "
        "Windows crashes with WinError 32. Executing 100% in RAM via io.BytesIO completely eliminates disk latency and file-locking errors.",
        body_style,
    ))

    # Summary Table
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Summary: How Module 1 Signals Feed the Neural Networks", h1_style))
    
    mapping_data = [
        [
            Paragraph("<b>Forensic Signal</b>", body_bold),
            Paragraph("<b>Physical Principle</b>", body_bold),
            Paragraph("<b>What Fraud Vector It Detects</b>", body_bold),
            Paragraph("<b>Target Consumer</b>", body_bold),
        ],
        [
            Paragraph("<b>Standardized RGB</b>", body_style),
            Paragraph("White alpha-compositing & EXIF transposition", body_style),
            Paragraph("Eliminates false border spikes and sideways scans", body_style),
            Paragraph("Stream 2 (ResNet-50) & Visual UI", body_style),
        ],
        [
            Paragraph("<b>In-Memory ELA Map</b>", body_style),
            Paragraph("JPEG Quantization Equilibrium delta", body_style),
            Paragraph("Altered amounts (INR 4k -> 14k), modified dates, inserted text", body_style),
            Paragraph("Stream 1 (EfficientNet-B0)", body_style),
        ],
        [
            Paragraph("<b>Noise Residual Matrix</b>", body_style),
            Paragraph("Non-linear 3x3 median filter subtraction", body_style),
            Paragraph("Erased stamps, removed text, AI inpainting (sterile patches)", body_style),
            Paragraph("Decision Fusion Brain & Red-Box Localizer", body_style),
        ],
    ]
    t_mapping = Table(mapping_data, colWidths=[1.5 * inch, 1.8 * inch, 2.0 * inch, 1.7 * inch])
    t_mapping.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_mapping)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {pdf_path.resolve()}")
    return pdf_path


if __name__ == "__main__":
    create_module1_pdf()
