# Module 9: Deterministic Arithmetic Logic Verifier & PDF Certificate Exporter

---

## 1. Executive Summary & Forensic Context

In forensic document auditing and fraud mitigation, automated systems face two fundamental failure modes:
1. **Perceptual False Positives / Negatives**: Deep neural networks can be tricked by high-quality raster inpainting, font-matching typography, and uniform JPEG re-quantization.
2. **Evidentiary Inadmissibility**: Probabilistic model outputs without cryptographic provenance or deterministic verification cannot withstand judicial scrutiny in insurance dispute litigation or civil court proceedings.

**Module 9** completes the analytical core of DocGuard by introducing a **Neuro-Symbolic Hybrid Architecture** and an automated **Court-Admissible PDF Certificate Exporter**:
* [`logic_checker.py`](file:///c:/Users/Harshit/Desktop/DocGuard/logic_checker.py): Extracts itemized monetary lines from financial documents and performs deterministic symbolic arithmetic auditing ($\sum \text{Line Items} + \text{Tax} - \text{Discount} \stackrel{?}{=} \text{Grand Total}$).
* [`report_generator.py`](file:///c:/Users/Harshit/Desktop/DocGuard/report_generator.py): Synthesizes findings across all 9 forensic modules into a publication-grade, two-page institutional PDF certificate complete with SHA-256 cryptographic provenance, tri-panel visual evidence montages, and a standardized 6-factor forensic scorecard.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           MODULE 9: DUAL-PIPELINE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   [Document Input: Native PDF or Scanned Raster Image]                                  │
│             │                                                                           │
│             ├──► [PyMuPDF Native Text Parser] ──► [Symbolic Arithmetic Engine]          │
│             │                                     - Currency & Numeric Tokenizer        │
│             │                                     - Line Item vs Total Classifier       │
│             │                                     - sum(Items) == Grand Total Audit     │
│             │                                     - Deterministic Discrepancy Delta (Δ) │
│             │                                                                           │
│             └──► [ReportLab Platypus Engine]  ──► [Certified Forensic PDF Report]       │
│                                                   - Cryptographic SHA-256 / MD5 Hash    │
│                                                   - Executive Verdict Banner & Score    │
│                                                   - Tri-Panel Visual Evidence Montage   │
│                                                   - Six-Factor Forensic Scorecard       │
│                                                   - FRE 902(14) Statutory Certification │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Neuro-Symbolic AI: Preventing Neural Hallucination

### 2.1 The Perceptual vs. Cognitive Divide
Modern deep neural networks (e.g. ResNet-50 and EfficientNet-B0) excel at **perceptual pattern recognition**:
* Detecting microscopic high-frequency edge seams.
* Identifying discrete cosine transform (DCT) quantization grid misalignments.
* Mapping continuous pixel distributions onto high-dimensional latent manifolds.

However, deep learning models are **stochastic approximators** that operate without symbolic grounding:
* A neural network has no internal representation of arithmetic axioms ($1,000 + 3,000 = 4,000$).
* When a fraudster alters an invoice's grand total from `INR 4,000` to `INR 14,000`, they often preserve the authentic font family, antialiasing halo, and raster compression of the original document. Under adverse conditions or heavy recompression, a purely neural detector might assign a borderline probability ($p = 0.48$), failing to trigger an alarm.

### 2.2 Symbolic Axiomatic Verification
In contrast, symbolic logic engines operate over discrete, deterministic domain rules:
$$\text{Expected Total} = \sum_{i=1}^N \text{Amount}_i + \text{Tax} - \text{Discount}$$
$$\Delta = \left| \text{Declared Grand Total} - \text{Expected Total} \right|$$

$$\text{Verdict} = \begin{cases} \text{ARITHMETIC\_CONSISTENT}, & \text{if } \Delta \le \epsilon \\ \text{ARITHMETIC\_TAMPERING\_DETECTED}, & \text{if } \Delta > \epsilon \end{cases}$$

Where $\epsilon = 0.01$ accounts for rounding in fractional pennies.

By combining neural visual perception with symbolic arithmetic verification:
1. **The Neural Stream** localizes *where* pixel tampering occurred.
2. **The Symbolic Stream** mathematically proves *why* the financial claims are fraudulent.
3. Neither system can hallucinate without being immediately checked by the other.

---

## 3. Cryptographic Provenance & Legal Evidentiary Admissibility

### 3.1 Statutory Standards: FRE 902(14) & ISO/IEC 27037
In legal adjudication (such as commercial contract enforcement or insurance claim denial), digital evidence must satisfy strict statutory hurdles:
* **Federal Rules of Evidence Rule 902(14)** allows self-authenticating digital records if they are accompanied by a certified process ensuring data integrity from the moment of acquisition.
* **ISO/IEC 27037:2012** prescribes guidelines for digital evidence handling, requiring an immutable, auditable **Chain of Custody**.

### 3.2 The Mathematical Mechanics of SHA-256 Hashing
DocGuard hashes the raw input byte stream at acquisition using the NIST-standardized **Secure Hash Algorithm (SHA-256)**:
$$\mathbf{H} = \text{SHA-256}(\mathbf{M}) \in \{0, 1\}^{256}$$

Key cryptographic properties:
1. **Collision Resistance**: It is computationally infeasible ($\mathcal{O}(2^{128})$ operations) to locate two distinct documents $\mathbf{M}_1 \neq \mathbf{M}_2$ such that $\text{SHA-256}(\mathbf{M}_1) = \text{SHA-256}(\mathbf{M}_2)$.
2. **The Avalanche Effect**: Flipping a single bit in an invoice (e.g. changing digit `'4'` to `'1'`) mutates more than $50\%$ of the output hexadecimal characters:
   * Authentic Invoice: `8A5B289F...`
   * Spliced Invoice:   `3CDD0E9D...`

Embedding this cryptographic fingerprint directly into the PDF certificate permanently binds the forensic report to that exact document binary.

---

## 4. Programmatic Architecture of `logic_checker.py`

[`logic_checker.py`](file:///c:/Users/Harshit/Desktop/DocGuard/logic_checker.py) implements automated financial text ingestion and ledger verification:

```
[Raw Lines / PDF] ──► [Currency Sanitizer] ──► [Classification RegEx] ──► [Ledger Audit]
```

### 4.1 Multi-Currency & Number Normalization
Document text displays currency amounts across differing conventions:
* Western numbering: `$14,000.00`
* South Asian numbering: `INR 1,40,000.00`
* Accounting negatives: `(500.00)` $\to -500.00$

The normalization pipeline uses compiled regular expressions:
```python
CURRENCY_PATTERN = r"(?:INR|Rs\.?|₹|USD|\$|EUR|€|GBP|£|CAD|AUD|JPY|¥)"
NUMBER_PATTERN = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?"
```
The sanitizer strips formatting punctuation, isolates the monetary magnitude, and standardizes currency denominations into IEEE 754 double-precision floats.

### 4.2 Document Line Classification
Lines are parsed and categorized into functional invoice roles:
* **Grand Total Anchors**: Matched against `RE_GRAND_TOTAL` (`grand total`, `total amount`, `net payable`, `amount due`).
* **Subtotal Anchors**: Matched against `RE_SUBTOTAL` (`subtotal`, `base amount`).
* **Tax Line Items**: Matched against `RE_TAX` (`tax`, `gst`, `vat`, `cgst`, `sgst`).
* **Discounts**: Matched against `RE_DISCOUNT` (`discount`, `rebate`, `credit`).
* **Itemized Line Items**: Non-metadata lines containing numerical entries that are accumulated into the active ledger tally.

---

## 5. Programmatic Architecture of `report_generator.py`

[`report_generator.py`](file:///c:/Users/Harshit/Desktop/DocGuard/report_generator.py) builds presentation-grade PDF certificates via ReportLab’s Platypus (Page Layout and Typography Using Scripts) layout engine.

### 5.1 Document Layout Geometry
The certificate is strictly formatted into a **two-page institutional briefing**:
* **Page Dimensions**: Standard Letter ($8.5 \times 11\text{ in}$ / $612 \times 792\text{ pt}$).
* **Margin Boundary**: Compact $36\text{ pt}$ ($0.5\text{ in}$) borders providing an active printable grid of $540 \times 720\text{ pt}$.
* **Typography**: Professional typography using Helvetica, Helvetica-Bold, and Courier (for cryptographic checksums).

### 5.2 Certificate Sections Breakdown

#### Page 1:
1. **Header Banner**: Deep Slate Navy background (`#0F172A`) with white title, certificate UUID (`DG-XXXXXXXX`), UTC timestamp, and statutory standards reference.
2. **Executive Verdict Banner**: Color-coded callout card:
   * Crimson Red (`#DC2626` / `#FEF2F2`) for `TAMPERED / FRAUD DETECTED`.
   * Emerald Green (`#16A34A` / `#F0FDF4`) for `AUTHENTIC / CLEAN VERIFIED`.
   * Prominent display of the master **Forgery Risk Score (0–100%)**.
3. **Legal Chain of Custody Table**:
   * File name, byte size, acquisition timestamp.
   * Full 64-character SHA-256 and MD5 cryptographic hashes in monospace Courier font.
4. **Tri-Panel Visual Forensic Evidence Montage**:
   * Synchronized side-by-side $172 \times 120\text{ pt}$ thumbnails:
     * *Panel A*: Raw RGB Document.
     * *Panel B*: In-Memory ELA Compression Delta Map.
     * *Panel C*: XAI Anomaly Map (Bounding boxes with translucent pink highlights).

#### Page 2:
5. **Six-Factor Forensic Scorecard**:
   * Systematic evaluation across all 6 forensic vectors:
     1. Deep Multimodal Fusion Brain
     2. Error Level Analysis (ELA)
     3. Spatial Noise Residuals
     4. PDF Binary Architecture (`%%EOF`)
     5. Font Seams & Typography
     6. Deterministic Arithmetic Ledger
   * Categorical badges (`PASS`, `ALERT`, `FAIL`) per factor.
6. **Symbolic Arithmetic Ledger Audit Table**:
   * Line items total vs declared grand total comparison with explicit discrepancy reporting.
7. **Forensic Adjudication Notes**: Custom observations detailing specific coordinates and anomalies.
8. **Statutory Evidentiary Seal**: Formal certification conforming to FRE 902(14) and ISO/IEC 27037 standards.

---

## 6. Comparative Forensic Verification Case Study

| Forensic Vector | Authentic Reference (`sample_invoice_pristine.pdf`) | Tampered Document (`sample_invoice_tampered.png`) |
|:---|:---:|:---:|
| **Consultation Fee** | `INR 1,000` | `INR 1,000` |
| **Prescription Medicines** | `INR 3,000` | `INR 3,000` |
| **Calculated Item Sum** | **`INR 4,000.00`** | **`INR 4,000.00`** |
| **Declared Grand Total** | **`INR 4,000.00`** | **`INR 14,000.00`** |
| **Arithmetic Discrepancy ($\Delta$)** | **`INR 0.00`** | **`INR +10,000.00`** |
| **Ledger Alarm** | `False` (Balanced) | **`True` (Alarm Triggered)** |
| **Neural Forgery Risk** | Safe | **`52.63%` (`FORGED`)** |
| **PDF Audit Certificate** | Available | **Exported ([`sample_tampered_audit_certificate.pdf`](file:///c:/Users/Harshit/Desktop/DocGuard/evaluation/sample_tampered_audit_certificate.pdf))** |
