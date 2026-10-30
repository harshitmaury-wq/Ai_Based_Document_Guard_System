# PROJECT REPORT & TECHNICAL BLUEPRINT
## DocGuard: AI-Powered Multimodal Document Forgery & Manipulation Detection Platform

---

### Project Specification Overview

| Project Attribute | Technical Specification |
|:---|:---|
| **Project Title** | **DocGuard**: Multimodal Document Forgery & Manipulation Detection Platform |
| **Domain** | Computer Vision, Digital Image Forensics, Deep Learning & Explainable AI (XAI) |
| **Primary Language** | Python 3.11+ (64-bit) |
| **Deep Learning Framework** | PyTorch 2.x, Torchvision (CUDA 12.1 acceleration) |
| **Target Hardware (Calibrated)** | **Intel/AMD 4 Cores / 8 Threads CPU**, **NVIDIA GeForce GTX 1650 (4GB VRAM)**, Windows 11 |
| **Dataset Strategy** | Real-World Public Benchmark Datasets (e.g. SROIE / Curated Invoices) into `dataset/real` and `dataset/forged` |
| **Core Architecture** | Staged Multi-Stream Transfer Learning: **Stream 1 (ELA EfficientNet-B0)** + **Stream 2 (RGB ResNet-50)** + **Stream 3 (Multimodal Fusion Brain)** |
| **Core Deliverables** | Dual-Stream Neural Models, XAI Localization Engine (Grad-CAM + OpenCV Contours), Streamlit Web Platform (Auditing + Analytics), Court-Grade PDF Audit Exporter, Arithmetic OCR Integrity Verifier |
| **Project Status** | **Architecture Approved & Hardware-Calibrated for GTX 1650 (Ready for Phased Execution)** |

---

## 1. Abstract & Executive Summary

Digital document fraud—specifically unauthorized alterations to invoices, medical receipts, academic certificates, tax filings, and insurance claims—causes billions of dollars in losses annually. With accessible digital photo editing software and generative AI tools, bad actors can alter financial numbers, dates, and names seamlessly without leaving visual flaws visible to the human eye.

Traditional verification systems require access to the original reference document or a central issuer database, which insurance companies, banks, and academic institutions rarely possess during initial claims submission.

**DocGuard** is a blind, reference-free digital document forensics platform. The system evaluates whether a document has been digitally altered without requiring access to the original version. It achieves this by combining:
1. **File-level binary forensics**: Inspecting PDF metadata, incremental updates, multiple `%%EOF` markers, and font structures.
2. **Physical compression and noise forensics**: In-memory Error Level Analysis (ELA) and high-pass noise residuals.
3. **Deep learning feature extraction**: A dual-stream network capturing compression discrepancies and visual boundary seams.
4. **Explainable AI (XAI)**: Grad-CAM activation heatmaps and OpenCV bounding-box anomaly localization.
5. **Deterministic arithmetic rules**: Preventing neural hallucination by mathematically validating that itemized sub-amounts sum to reported totals.

The platform produces two core outputs: a **Calibrated Forgery Risk Score (0–100%)** and a **Pixel-Accurate Visual Localization Map** highlighting the exact coordinates of tampered text.

The architecture is specifically engineered to train and infer efficiently within the memory constraints of an **NVIDIA GeForce GTX 1650 (4GB VRAM)** and a **4-Core / 8-Thread CPU** on Windows 11 using staged transfer learning, layer freezing, and in-memory byte streaming. All training and validation metrics (Loss, Accuracy, Precision, Recall, F1-Score, and Confusion Matrix) are automatically logged and plotted for post-experiment academic analysis and project defense.

---

## 2. Hardware Calibration: Adapting to GTX 1650 (4GB VRAM) & 4-Core CPU

The original specification document referenced an RTX 3050 (6GB VRAM). We have recalibrated the entire pipeline for your confirmed hardware:
* **GPU**: NVIDIA GeForce GTX 1650 (4,096 MiB VRAM / Turing TU117 / 50W TGP)
* **CPU**: 4 Cores / 8 Threads
* **RAM & OS**: Windows 11 64-bit

### GPU Memory & Performance Optimization Strategy

1. **Input Dimensions ($224 \times 224$)**: Standard ImageNet dimensions preserve fine edge features while keeping activation tensor memory under **1.9 GB VRAM** during backpropagation, well below the 4.0 GB limit.
2. **Layer Freezing**:
   - In Phase 1 (EfficientNet-B0), all foundational convolutional layers are frozen (`requires_grad = False`). Only the 64-dim classification head is trained.
   - In Phase 2 (ResNet-50), Layers 1 through 3 are frozen. Only the classification head trains.
   - In Phase 3 (Fusion), only Layer 4 of ResNet-50 and Block 7 of EfficientNet-B0 are fine-tuned alongside the fusion classifier head.
3. **DataLoader Multiprocessing on Windows**: With 4 cores / 8 threads, setting `num_workers = 2` with `pin_memory = True` prevents Windows process spawning bottlenecks and socket stalls.
4. **Staged Memory Allocation Schedule**:

| Phase | Script | Target Network | Batch Size | Epochs | GTX 1650 Est. Time | Peak VRAM |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Phase 1** | `phase1_ela.py` | ELA Classifier Head (EfficientNet-B0) | 8 | 5 | ~15–20 mins | **~1.2 GB** |
| **Phase 2** | `phase2_rgb.py` | RGB Classifier Head (ResNet-50) | 8 | 5 | ~18–22 mins | **~1.4 GB** |
| **Phase 3** | `phase3_fusion.py`| Master Fusion Head + Top Blocks | 4 | 5 | ~20–25 mins | **~1.8 GB** |
| **Inference**| `predict.py` / `app.py` | Single Document End-to-End | 1 | — | **< 150 ms** | **~0.9 GB** |

---

## 3. The 6 Forensic Inspection Factors

1. **Factor 1: Differential Compression Discrepancies (ELA)**: Re-saves the image in RAM at Quality 95 and computes absolute pixel difference $\Delta(x, y) = |I_{\text{original}}(x, y) - I_{\text{resaved}}(x, y)|$. Original text remains dark; tampered text glows with an amplified error rate.
2. **Factor 2: Noise Texture & Paper Grain Variance**: Authentic scanned documents possess uniform high-frequency sensor noise and paper grain. Median filter subtraction $R(x, y) = |I(x, y) - \text{MedianFilter}(I(x, y), 3)|$ extracts noise residuals to flag sudden drops or spikes in local variance.
3. **Factor 3: Visual Edge Seams & Anti-Aliasing Halos**: Pasting a copied patch or retyping numbers introduces boundary anti-aliasing seams and sub-pixel color feathering, evaluated by convolutional kernels in ResNet-50 (Stream 2).
4. **Factor 4: Font Geometry, Kerning & Baseline Alignment**: Inserted digits rarely align with the original font template (e.g. an altered "1" sitting 1–2 pixels above the horizontal baseline or having a mismatched stroke width).
5. **Factor 5: $8 \times 8$ JPEG Block Grid Periodicity**: JPEG encoding partitions images into strict $8 \times 8$ pixel DCT blocks. Spliced patches almost never land on an exact multiple of 8 pixels, creating a phase mismatch in the Periodic Block Artifact Grid (BAG).
6. **Factor 6: Arithmetic & Semantic Logic Verification**: Deterministic rule engine verifying $\sum \text{Itemized Amounts} \stackrel{?}{=} \text{Grand Total}$. Purely mathematical validation with zero neural hallucination.

---

## 4. File Organization & 11 Implementation Modules

### Workspace Directory Structure
```text
DocGuard/
├── dataset/
│   ├── real/                     # 100% genuine real-world scanned documents
│   └── forged/                   # Real digitally tampered & modified documents
│
├── evaluation/                   # AUTO-GENERATED: Plots & metrics for defense
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── metrics_summary.json
│
├── preprocessor.py               # Module 1: In-memory ELA & noise extraction
├── pdf_forensics.py              # Module 2: Native PDF binary & metadata auditor
├── dataset_loader.py             # Module 3: PyTorch data loaders & stratified split
├── phase1_ela.py                 # Module 4: Train ELA Stream (EfficientNet-B0)
├── phase2_rgb.py                 # Module 5: Train RGB Stream (ResNet-50)
├── phase3_fusion.py              # Module 6: Train Master Fusion Brain
├── evaluate_model.py             # Module 7: Generates & saves all presentation plots
├── predict.py                    # Module 8: Single-document inference & red-box localizer
├── logic_checker.py              # Module 9: Arithmetic OCR integrity verifier
├── report_generator.py           # Module 10: Court-grade PDF audit certificate exporter
├── app.py                        # Module 11: Streamlit web platform (Auditing + Analytics)
│
├── PROJECT_EXECUTION_LOG.md      # Continuous living long-format engineering log
├── requirements.txt              # Dependencies list
├── PROJECT_PROPOSAL.md           # Original proposal reference
└── DOCGUARD_PROJECT_REPORT.md    # Complete academic project documentation
```
