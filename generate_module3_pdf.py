#!/usr/bin/env python3
"""
DocGuard Module 3 Explanation PDF Generator
Compiles a publication-quality pedagogical report from MODULE_3_EXPLANATION.md
into MODULE_3_EXPLANATION.pdf using ReportLab.
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
            self.drawString(40, 760, "DOCGUARD FORENSIC ARCHITECTURE | MODULE 3: DATASET PIPELINE & STRATIFIED SPLITTER")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(40, 752, 572, 752)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(40, 42, 572, 42)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(40, 30, "DocGuard Digital Forensics Platform | Confidential & Academic Architecture")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_str)
        self.restoreState()


def create_module3_pdf(output_pdf_path: str = "MODULE_3_EXPLANATION.pdf"):
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
    story.append(Paragraph("DocGuard — Module 3 Pedagogical & Technical Guide", title_style))
    story.append(Paragraph("Understanding the Real Dataset Loader & Stratified Splitter (dataset_loader.py)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceBefore=0, spaceAfter=12))

    # Section 1
    story.append(Paragraph("1. What is Module 3 Doing? (The Objective in Simple Language)", h1_style))
    p1 = (
        "Imagine you want to train an intelligent forensic investigator (a deep neural network) to detect altered "
        "medical invoices, tax receipts, and bank statements.<br/><br/>"
        "To teach an AI model effectively, you need thousands of document examples divided into two categories: "
        "<b>Authentic (Genuine)</b> and <b>Forged (Tampered)</b>. But machine learning models do not learn directly from "
        "raw image files scattered in random desktop folders. Neural networks speak only one language: "
        "<b>multidimensional arrays of numbers (mathematical tensors)</b>.<br/><br/>"
        "Furthermore, how you feed those documents into the AI determines whether it succeeds in the real world or fails completely:<br/>"
        "• If you feed the AI 90% authentic documents and only 10% forged documents, it becomes lazy and learns to guess 'authentic' every time.<br/>"
        "• If you train the AI only on clean synthetic computer graphics, it gets confused when it encounters real-world documents with scanner dust, folds, and paper textures.<br/>"
        "• If a document from the training set accidentally leaks into the validation set, the AI merely memorizes the answer like a cheating student rather than learning real forensic reasoning.<br/><br/>"
        "<b>Module 3 is the intelligent logistics coordinator and data pipeline that solves these fundamental challenges.</b>"
    )
    story.append(Paragraph(p1, body_style))

    # Callout Box: Core Mission
    callout_data = [[
        Paragraph(
            "<b>The Core Operational Mission of Module 3:</b><br/>"
            "• <b>Real-World Document Ingestion:</b> Automatically scans and organizes genuine documents into <code>dataset/real/</code> and tampered documents into <code>dataset/forged/</code>.<br/>"
            "• <b>Stratified 80/20 Partitioning:</b> Mathematically splits the data into 80% Training and 20% Validation while strictly guaranteeing identical class proportions in both sets.<br/>"
            "• <b>Dual-Stream Tensor Packaging:</b> For every document, outputs synchronized RGB image tensors and in-memory ELA compression maps $(3, 224, 224)$.<br/>"
            "• <b>Forensic-Safe Augmentations:</b> Applies subtle micro-rotations ($\pm 3^\circ$) and lighting jitter while strictly avoiding destructive horizontal flips that reverse text.<br/>"
            "• <b>Hardware-Calibrated GPU Streaming:</b> Batches documents (8 at a time) using direct page-locked DMA memory streaming, keeping peak VRAM below 1.4 GB on the GTX 1650.",
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
    story.append(Paragraph("2. How Does Module 3 Achieve This? (The 4-Stage Data Pipeline)", h1_style))
    story.append(Paragraph(
        "Module 3 bridges raw image files and GPU tensor operations through a 4-stage pipeline:",
        body_style
    ))

    pipeline_items = [
        ("Stage 1: Multi-Format Recursive Scanner",
         "Traverses root directories to ingest all supported formats (.png, .jpg, .jpeg, .tiff). Maps ground truth labels: "
         "0 for Authentic/Real and 1 for Forged/Tampered, with automated folder verification."),
        ("Stage 2: Stratified 80/20 Sampling Partition",
         "Calculates the exact population class ratio and enforces that the 80% Training partition and 20% Validation partition "
         "contain identical class ratios. Mathematically guarantees zero data leakage between partitions (disjoint sets)."),
        ("Stage 3: Dual-Stream PyTorch Dataset (DocGuardDataset)",
         "Generates two parallel tensors for every document: Stream 1 (in-memory ELA compression delta map at Q=95 via Module 1) "
         "and Stream 2 (normalized RGB document image), scaled to (3, 224, 224) and standardized via ImageNet statistics."),
        ("Stage 4: Hardware-Calibrated DataLoader Factory",
         "Assembles mini-batches of 8 documents. Uses num_workers=2 to prevent Windows multiprocessing socket stalls, "
         "and enables CUDA memory pinning (pin_memory=True) for zero-copy DMA streaming to the GTX 1650 GPU.")
    ]

    for stage_title, stage_desc in pipeline_items:
        story.append(Paragraph(f"<b>• {stage_title}:</b> {stage_desc}", body_style))

    story.append(Spacer(1, 10))

    # Section 3
    story.append(Paragraph("3. Comprehensive Glossary of Technical & ML Concepts", h1_style))

    # Group A
    story.append(Paragraph("A. Dataset Architecture & Domain Adaptation", h2_style))

    story.append(Paragraph("1. The Synthetic-to-Real Domain Gap", h3_style))
    story.append(Paragraph(
        "The performance drop that occurs when a machine learning model is trained exclusively on synthetic computer-generated "
        "images and deployed on real-world scanned documents. Synthetic documents have unnaturally sharp, pixel-perfect boundaries "
        "and zero sensor noise. Authentic scanned documents contain optical lens distortion, microscopic paper grain, ink bleed, "
        "creases, and scanner illumination falloff. Training on real-world scanned documents ensures the convolutional filters "
        "learn organic forensic features rather than overfitting to synthetic artifacts.",
        body_style
    ))

    story.append(Paragraph("2. Binary Ground Truth Labeling (0 vs. 1)", h3_style))
    story.append(Paragraph(
        "The mathematical encoding of document authenticity: <b>Y = 0</b> represents Authentic / Real (pristine documents issued "
        "by legitimate institutions), and <b>Y = 1</b> represents Forged / Tampered (documents containing digital text alterations, "
        "spliced figures, or erased stamps). These scalars form the target vectors for neural network loss functions.",
        body_style
    ))

    # Group B
    story.append(Paragraph("B. Statistical Partitioning & Data Splitting", h2_style))

    story.append(Paragraph("1. The 80/20 Train/Validation Split Rule", h3_style))
    story.append(Paragraph(
        "The standard practice of dividing data into a Training Set (80%), which is actively shown to the neural network during "
        "backpropagation weight updates, and a Validation Set (20%), which is held out to measure true generalization ability. "
        "The validation set acts as an honest benchmark to detect when the model begins to overfit.",
        body_style
    ))

    story.append(Paragraph("2. Stratified Sampling vs. Naive Random Splitting", h3_style))
    story.append(Paragraph(
        "In naive random splitting, documents are assigned to train or validation purely by chance, which often introduces an "
        "accidental class imbalance on moderate datasets (e.g. 70% real in training, but only 30% in validation). "
        "Stratified sampling computes the exact proportion of positive samples in the full population and guarantees that both "
        "partitions maintain this exact ratio: <code>P(Y = 1 | Train) == P(Y = 1 | Val) == P(Y = 1 | Full)</code>. "
        "This eliminates distribution drift and keeps decision thresholds calibrated.",
        body_style
    ))

    story.append(Paragraph("3. Data Leakage Prevention (Disjoint Partitions)", h3_style))
    story.append(Paragraph(
        "The severe experimental error where information from the validation set contaminates the training set. "
        "If a document appears in both sets, the network merely memorizes the specific image rather than learning general forensic rules. "
        "DocGuard enforces strict disjointness: <code>Train ∩ Val = ∅</code>.",
        body_style
    ))

    # Group C
    story.append(Paragraph("C. Forensic-Safe Data Augmentation", h2_style))

    story.append(Paragraph("1. Why Standard CV Augmentations Fail on Documents", h3_style))
    story.append(Paragraph(
        "In general object recognition (e.g. cats vs. dogs), horizontal flips and heavy blurs are standard. In document forensics, "
        "flipping an image horizontally reverses text characters into illegible mirror-images, corrupting typography kernels. "
        "Similarly, heavy blurring destroys the subtle JPEG quantization equilibrium signals that Error Level Analysis (ELA) "
        "requires to spot modified digits.",
        body_style
    ))

    story.append(Paragraph("2. Forensic-Safe Augmentation Policy", h3_style))
    story.append(Paragraph(
        "DocGuard applies domain-specific micro-augmentations: <b>Micro-Rotation ($\pm 3^\circ$)</b> with white paper border fill "
        "simulates slight skew from physical scanner feed rollers, and <b>Micro-Photometric Jitter ($\pm 5\%$)</b> simulates "
        "scanner lamp brightness variations without altering frequency-domain compression grids.",
        body_style
    ))

    # Group D
    story.append(Paragraph("D. Tensor Representations & Normalization", h2_style))

    story.append(Paragraph("1. Dual-Stream Tensors (RGB + ELA)", h3_style))
    story.append(Paragraph(
        "DocGuard produces two complementary tensor representations of every document:<br/>"
        "• <b>Stream 1 Tensor (ELA):</b> Shape (3, 224, 224). Captures in-memory JPEG quantization error deltas computed at Q=95. "
        "Altered numbers glow bright white.<br/>"
        "• <b>Stream 2 Tensor (RGB):</b> Shape (3, 224, 224). Captures raw visual layout, character kerning, and edge seams.<br/>"
        "Feeding both streams simultaneously enables the multimodal brain to cross-reference visual alignment against compression age.",
        body_style
    ))

    story.append(Paragraph("2. ImageNet Distribution Normalization (z-Score Scaling)", h3_style))
    story.append(Paragraph(
        "Pre-trained deep neural networks require input tensors normalized using standard ImageNet distribution constants: "
        "<code>x_norm = (x - μ) / σ</code> with <code>μ = [0.485, 0.456, 0.406]</code> and <code>σ = [0.229, 0.224, 0.225]</code>. "
        "This prevents gradient vanishing or explosion and ensures pre-trained weights receive inputs in their optimal numerical range.",
        body_style
    ))

    # Group E
    story.append(Paragraph("E. Hardware Calibration & PyTorch DataLoaders", h2_style))

    story.append(Paragraph("1. Batch Size Calibration for 4 GB VRAM", h3_style))
    story.append(Paragraph(
        "On the NVIDIA GeForce GTX 1650 (4,096 MiB VRAM), large batch sizes trigger CUDA Out-of-Memory crashes. "
        "Setting <code>batch_size = 8</code> restricts peak activation memory to approximately 1.4 GB, leaving ample headroom "
        "for model parameters, gradients, optimizer momentum buffers, and OS display drivers.",
        body_style
    ))

    story.append(Paragraph("2. Multiprocessing Workers (num_workers = 2)", h3_style))
    story.append(Paragraph(
        "On Windows 11, worker processes are created via 'spawn' rather than 'fork'. Setting <code>num_workers = 2</code> on our "
        "4-core CPU ensures that background CPU workers load and preprocess the next batch in RAM while the GPU processes the current batch, "
        "maximizing GPU utilization without socket stalls.",
        body_style
    ))

    story.append(Paragraph("3. CUDA Memory Pinning (pin_memory = True)", h3_style))
    story.append(Paragraph(
        "Allocates CPU host memory as page-locked (pinned), preventing the Windows virtual memory manager from swapping it to disk. "
        "This enables direct, high-speed Direct Memory Access (DMA) transfers across the PCIe bus to GPU VRAM without CPU intervention.",
        body_style
    ))

    # Summary Table
    story.append(Spacer(1, 6))
    story.append(Paragraph("Summary: How Module 3 Connects Data to Neural Streams", h2_style))

    summary_data = [
        [Paragraph("Pipeline Stage", table_header),
         Paragraph("Core Responsibility", table_header),
         Paragraph("Technical Mechanism", table_header),
         Paragraph("Downstream Consumer", table_header)],
        [Paragraph("Path Ingestion", table_cell_bold),
         Paragraph("Discovers document images", table_cell),
         Paragraph("Recursive scan across 5 extensions (.png, .jpg, .tiff)", table_cell),
         Paragraph("Stratified Splitter", table_cell)],
        [Paragraph("Stratified Splitter", table_cell_bold),
         Paragraph("Guarantees class balance", table_cell),
         Paragraph("Scikit-Learn train_test_split(stratify=y, test_size=0.2)", table_cell),
         Paragraph("Dataset Wrappers", table_cell)],
        [Paragraph("DocGuardDataset", table_cell_bold),
         Paragraph("Produces dual-stream tensors", table_cell),
         Paragraph("Module 1 ELA engine + Torchvision transforms", table_cell),
         Paragraph("PyTorch DataLoaders", table_cell)],
        [Paragraph("Forensic Augmentation", table_cell_bold),
         Paragraph("Prevents visual overfitting", table_cell),
         Paragraph("Micro-rotation (+/- 3 deg) with white paper border fill", table_cell),
         Paragraph("Training Pipeline", table_cell)],
        [Paragraph("DataLoader Factory", table_cell_bold),
         Paragraph("Hardware-calibrated streaming", table_cell),
         Paragraph("Batch size 8, pin_memory=True, num_workers=2", table_cell),
         Paragraph("Phases 1, 2, 3 Neural Training", table_cell)],
    ]

    summary_table = Table(summary_data, colWidths=[110, 110, 192, 120])
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
    create_module3_pdf()
