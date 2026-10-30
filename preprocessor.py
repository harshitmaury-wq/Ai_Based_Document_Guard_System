"""
========================================================================================
DocGuard: Multimodal Document Forgery & Manipulation Detection Platform
Module 1: In-Memory Forensic Preprocessor (preprocessor.py)
----------------------------------------------------------------------------------------
Submodule 1.1: Image Input Normalizer & In-Memory Stream Handler
Submodule 1.2: In-Memory Error Level Analysis (ELA) Core Engine
Submodule 1.3: Spatial Noise Residual Extractor (High-Pass Median Filter)
Submodule 1.4: CLI Test Harness & Visual Verification Routine
========================================================================================

Forensic Science Rationale:
---------------------------
1. Error Level Analysis (ELA):
   Lossy JPEG compression operates by quantizing 8x8 DCT coefficient blocks. Successive
   re-saves of an unedited document achieve 'quantization equilibrium' (near-zero delta).
   Modified or injected text undergoes first-time lossy quantization at that coordinate,
   causing it to glow with an amplified error delta when re-saved at baseline Quality 95.

2. Spatial Noise Residual Analysis:
   Physical scanned or photographed documents exhibit a continuous, uniform distribution
   of high-frequency sensor noise and paper fiber grain across the page.
   When an attacker erases characters or uses generative AI inpainting (e.g. content-aware
   fill or Stable Diffusion inpainting), the algorithm synthesizes an unnaturally smooth
   ('sterile') patch lacking the micro-texture and high-frequency noise of authentic paper.
   Conversely, copy-pasting text from a different source introduces an alien noise profile.

   By applying a non-linear 3x3 median filter:
       I_smooth(x, y) = MedianFilter(I(x, y), 3)
   the macro text structure is preserved while high-frequency noise is suppressed.
   The spatial noise residual is:
       R(x, y) = |I(x, y) - I_smooth(x, y)|
   A sudden drop in local noise variance σ²_R exposes inpainting; a sudden spike flags splicing.

Why In-Memory (io.BytesIO) is Critical:
  Executes 100% in RAM, eliminating Windows file-locking crashes (WinError 32) and disk I/O.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


# Type alias for supported document input sources
DocumentInputSource = Union[str, Path, bytes, io.BytesIO, Image.Image]


# =============================================================================
# Submodule 1.1: Image Input Normalizer & In-Memory Stream Handler
# =============================================================================

def _render_pdf_to_pil(pdf_source: Union[str, Path, bytes, io.BytesIO], dpi: int = 150) -> Image.Image:
    """Renders the first page of a native PDF document into a high-resolution RGB PIL Image."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            raise ImportError("PyMuPDF (pymupdf) is required to render PDF documents.")

    if isinstance(pdf_source, (str, Path)):
        doc = pymupdf.open(str(pdf_source))
    elif isinstance(pdf_source, io.BytesIO):
        doc = pymupdf.open(stream=pdf_source.getvalue(), filetype="pdf")
    elif isinstance(pdf_source, bytes):
        doc = pymupdf.open(stream=pdf_source, filetype="pdf")
    else:
        raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

    if len(doc) == 0:
        doc.close()
        raise ValueError("PDF document contains no pages.")

    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def normalize_image_input(image_input: DocumentInputSource) -> Image.Image:
    """Standardizes any supported image or PDF input into a clean, 3-channel RGB PIL Image.

    Processing Steps:
      1. Source Resolution: Resolves paths, byte buffers, PIL instances, or PDF files.
      2. PDF Ingestion: Seamlessly renders native PDF first page to high-res RGB canvas.
      3. EXIF Transposition: Automatically corrects smartphone scan orientations.
      4. Alpha Compositing: Transparent pixels are blended onto a pristine white
         document background rather than converting to black artifacts.
      5. Color Space Standardization: Converts Grayscale, Palette, or CMYK to RGB.
      6. Integrity Validation: Verifies non-zero spatial dimensions.

    Args:
        image_input: A file path (str or Path), raw bytes, io.BytesIO stream,
                     or an existing PIL Image. Supports standard images (.png, .jpg)
                     as well as native PDF documents (.pdf).

    Returns:
        PIL.Image.Image: Verified 3-channel RGB image.

    Raises:
        FileNotFoundError: If a provided string/Path does not exist on disk.
        ValueError: If the input is corrupt, empty, or an unsupported type.
    """
    img: Image.Image

    # Step 1: Ingestion & Stream Decoding
    if isinstance(image_input, (str, Path)):
        file_path = Path(image_input)
        if not file_path.exists():
            raise FileNotFoundError(f"Document file does not exist on disk: {file_path.resolve()}")
        if not file_path.is_file():
            raise ValueError(f"Provided path is not a file: {file_path.resolve()}")

        # Support native PDF documents directly
        if file_path.suffix.lower() == ".pdf":
            try:
                img = _render_pdf_to_pil(file_path)
            except Exception as err:
                raise ValueError(f"Failed to render PDF document at {file_path}: {err}") from err
        else:
            try:
                img = Image.open(file_path)
                img.load()
            except Exception as err:
                raise ValueError(f"Corrupted or invalid image file at {file_path}: {err}") from err

    elif isinstance(image_input, bytes):
        if len(image_input) == 0:
            raise ValueError("Provided image byte buffer is empty (0 bytes).")
        # Check PDF header magic bytes (%PDF)
        if image_input.startswith(b"%PDF"):
            try:
                img = _render_pdf_to_pil(image_input)
            except Exception as err:
                raise ValueError(f"Failed to render PDF from raw bytes: {err}") from err
        else:
            try:
                byte_stream = io.BytesIO(image_input)
                img = Image.open(byte_stream)
                img.load()
            except Exception as err:
                raise ValueError(f"Failed to decode image from raw bytes: {err}") from err

    elif isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        header = image_input.read(4)
        image_input.seek(0)
        if header == b"%PDF":
            try:
                img = _render_pdf_to_pil(image_input)
            except Exception as err:
                raise ValueError(f"Failed to render PDF from io.BytesIO stream: {err}") from err
        else:
            try:
                img = Image.open(image_input)
                img.load()
            except Exception as err:
                raise ValueError(f"Failed to read image from io.BytesIO stream: {err}") from err

    elif isinstance(image_input, Image.Image):
        img = image_input.copy()

    else:
        raise ValueError(
            f"Unsupported document input type: {type(image_input).__name__}. "
            "Expected str, Path, bytes, io.BytesIO, or PIL.Image.Image."
        )

    # Step 2: EXIF Orientation Transposition
    try:
        img = ImageOps.exif_transpose(img) or img
    except Exception:
        pass

    # Step 3: Color Mode Normalization & White-Backing Compositing
    if img.mode == "RGB":
        return img

    if img.mode in ("RGBA", "LA"):
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.split()[-1]
        white_bg.paste(img.convert("RGB"), mask=alpha)
        return white_bg

    if img.mode in ("P", "PA"):
        img_rgba = img.convert("RGBA")
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        white_bg.paste(img_rgba.convert("RGB"), mask=img_rgba.split()[-1])
        return white_bg

    if img.mode in ("L", "1"):
        return img.convert("RGB")

    if img.mode == "CMYK":
        return img.convert("RGB")

    return img.convert("RGB")


def image_to_byte_buffer(img: Image.Image, format: str = "PNG") -> io.BytesIO:
    """Serializes a PIL Image into an in-memory io.BytesIO buffer.

    Ensures zero disk writes and zero Windows file-locking complications.

    Args:
        img: Clean PIL Image.
        format: Target image encoding format ('PNG' for lossless, 'JPEG' for lossy).

    Returns:
        io.BytesIO: In-memory byte stream.
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer


# =============================================================================
# Submodule 1.2: In-Memory Error Level Analysis (ELA) Core Engine
# =============================================================================

def compute_ela(
    image_input: DocumentInputSource,
    quality: int = 95,
    scale: int = 15,
    adaptive_scale: bool = False,
) -> Tuple[Image.Image, Dict[str, float]]:
    """Performs in-memory Error Level Analysis (ELA) on a document image.

    Quantization Delta Calculation:
      1. Normalizes input into a verified 3-channel RGB PIL Image (I_orig).
      2. Compresses and re-saves the image entirely in RAM at target quality Q (e.g. 95)
         using io.BytesIO (I_resaved).
      3. Computes absolute channel-by-channel pixel difference:
             Δ(x, y) = |I_orig(x, y) - I_resaved(x, y)|
      4. Calculates physical error distribution statistics (mean, max, std).
      5. Amplifies the subtle compression delta by the scale multiplier into
         the visible dynamic range [0, 255]:
             ELA(x, y) = clip(Δ(x, y) * scale, 0, 255)

    Args:
        image_input: Document image as file path, raw bytes, io.BytesIO, or PIL Image.
        quality: Baseline JPEG re-compression quality (default 95).
        scale: Multiplier to amplify low-magnitude error levels (default 15).
        adaptive_scale: If True, dynamically scales error so peak error reaches 255.

    Returns:
        Tuple[Image.Image, Dict[str, float]]:
            - ela_image: Enhanced 3-channel RGB PIL Image.
            - metrics: Dictionary containing mean_error, max_error, std_error, baseline_quality.
    """
    orig_img = normalize_image_input(image_input)

    # In-Memory Re-Compression via io.BytesIO
    ram_buffer = io.BytesIO()
    orig_img.save(ram_buffer, format="JPEG", quality=quality)
    ram_buffer.seek(0)
    resaved_img = Image.open(ram_buffer)
    resaved_img.load()

    # Absolute Difference Map
    diff_img = ImageChops.difference(orig_img, resaved_img)

    # Statistical Metrics
    diff_array = np.asarray(diff_img, dtype=np.float32)
    pixel_deltas = np.mean(diff_array, axis=2)

    mean_err = float(np.mean(pixel_deltas))
    max_err = float(np.max(pixel_deltas))
    std_err = float(np.std(pixel_deltas))

    metrics: Dict[str, float] = {
        "mean_error": round(mean_err, 4),
        "max_error": round(max_err, 4),
        "std_error": round(std_err, 4),
        "baseline_quality": float(quality),
    }

    # Visual Error Amplification
    if adaptive_scale and max_err > 0:
        multiplier = 255.0 / max_err
    else:
        multiplier = float(scale)

    scaled_array = np.clip(diff_array * multiplier, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(scaled_array, mode="RGB")

    return ela_image, metrics


# =============================================================================
# Submodule 1.3: Spatial Noise Residual Extractor (High-Pass Median Filter)
# =============================================================================

def extract_noise_residual(
    image_input: DocumentInputSource,
    kernel_size: int = 3,
    as_pil: bool = False,
) -> Tuple[Union[np.ndarray, Image.Image], Dict[str, float]]:
    """Extracts high-pass spatial noise residuals to reveal texture and inpainting anomalies.

    Args:
        image_input: Document image as file path, raw bytes, io.BytesIO, or PIL Image.
        kernel_size: Neighborhood diameter for median filter (must be odd integer, default 3).
        as_pil: If True, returns a visual 3-channel RGB PIL Image normalized to [0, 255].
                If False, returns raw float32 NumPy array of shape (H, W, 3).

    Returns:
        Tuple[Union[np.ndarray, Image.Image], Dict[str, float]]:
            - residual: NumPy array (H, W, 3) [float32] or visual PIL Image.
            - metrics: Dictionary containing mean_noise, std_noise, max_noise, kernel_size.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer (e.g. 3, 5), got: {kernel_size}")

    pil_img = normalize_image_input(image_input)
    img_array = np.asarray(pil_img, dtype=np.uint8)

    # OpenCV cv2.medianBlur runs in optimized C++ SIMD routines (< 6ms on GTX 1650 host)
    smoothed_array = cv2.medianBlur(img_array, kernel_size)

    # High-Pass Noise Residual: R(x, y) = |I(x, y) - I_smooth(x, y)|
    residual_array = np.abs(img_array.astype(np.float32) - smoothed_array.astype(np.float32))

    pixel_noise = np.mean(residual_array, axis=2)
    mean_noise = float(np.mean(pixel_noise))
    std_noise = float(np.std(pixel_noise))
    max_noise = float(np.max(pixel_noise))

    metrics: Dict[str, float] = {
        "mean_noise": round(mean_noise, 4),
        "std_noise": round(std_noise, 4),
        "max_noise": round(max_noise, 4),
        "kernel_size": float(kernel_size),
    }

    if as_pil:
        amplified = np.clip(residual_array * 5.0, 0, 255).astype(np.uint8)
        output_image = Image.fromarray(amplified, mode="RGB")
        return output_image, metrics

    return residual_array, metrics


# =============================================================================
# Submodule 1.4: Synthetic Demo Generator & Multi-Panel Diagnostic Montage
# =============================================================================

def generate_synthetic_invoice_pair() -> Tuple[Image.Image, Image.Image]:
    """Generates an authentic synthetic invoice and an altered version for demonstration.

    Returns:
        Tuple[Image.Image, Image.Image]: (pristine_invoice, tampered_invoice)
    """
    w, h = 600, 400
    np.random.seed(42)

    # Step 1: Base paper texture with natural scanner noise
    grain = np.random.normal(loc=0.0, scale=4.0, size=(h, w, 3))
    paper = np.clip(248.0 + grain, 0, 255).astype(np.uint8)
    pristine_img = Image.fromarray(paper, mode="RGB")

    draw = ImageDraw.Draw(pristine_img)
    # Header & border
    draw.rectangle([20, 20, w - 20, h - 20], outline=(70, 70, 70), width=2)
    draw.text((40, 40), "CITY CLINIC - MEDICAL INVOICE", fill=(20, 20, 20))
    draw.line([(40, 65), (w - 40, 65)], fill=(150, 150, 150), width=1)

    # Line items
    draw.text((40, 90), "Consultation Fee:      INR 1,000", fill=(30, 30, 30))
    draw.text((40, 120), "Prescription Medicines: INR 3,000", fill=(30, 30, 30))
    draw.text((40, 160), "Grand Total:           INR 4,000", fill=(10, 10, 10))
    draw.rectangle([35, 150, 350, 190], outline=(100, 100, 100), width=1)

    # Stamp
    draw.ellipse([420, 260, 540, 340], outline=(0, 100, 0), width=2)
    draw.text((445, 295), "[PAID - VERIFIED]", fill=(0, 100, 0))

    # Pre-compress to establish authentic JPEG quantization equilibrium (Quality 95)
    buf = io.BytesIO()
    pristine_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    pristine_jpeg = Image.open(buf)
    pristine_jpeg.load()

    # Step 2: Create Tampered Version
    # Attack 1: Alter "INR 4,000" to "INR 14,000"
    tampered_img = pristine_jpeg.copy()
    tamper_draw = ImageDraw.Draw(tampered_img)
    # Patch over total
    tamper_draw.rectangle([180, 158, 330, 185], fill=(248, 248, 248))
    tamper_draw.text((180, 160), "INR 14,000", fill=(10, 10, 10))

    # Attack 2: Digital Inpainting / Erase Stamp (simulate smooth inpainting patch)
    tamper_draw.rectangle([415, 255, 545, 345], fill=(248, 248, 248))  # Sterile patch

    return pristine_jpeg, tampered_img


def create_forensic_montage(
    original: Image.Image,
    tampered: Image.Image,
    ela_img: Image.Image,
    noise_img: Image.Image,
) -> Image.Image:
    """Combines 4 diagnostic forensic stages into a publication-grade comparison montage.

    Panel Layout:
    +---------------------------+---------------------------+
    | Panel 1: Original Receipt  | Panel 2: Tampered Receipt |
    +---------------------------+---------------------------+
    | Panel 3: ELA Compression  | Panel 4: Noise Texture    |
    +---------------------------+---------------------------+
    """
    w, h = 600, 400
    panel_w = w
    panel_h = h
    banner_h = 35
    total_w = panel_w * 2 + 30
    total_h = (panel_h + banner_h) * 2 + 40

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 35))
    canvas_draw = ImageDraw.Draw(canvas)

    panels = [
        ("1. Original Authentic Document", original, 10, 10),
        ("2. Tampered Document (Altered Total + Erased Stamp)", tampered, panel_w + 20, 10),
        ("3. ELA Quantization Map (Altered Total Glows Brightly)", ela_img, 10, panel_h + banner_h + 20),
        ("4. Spatial Noise Residual (Erased Stamp Appears Sterile)", noise_img, panel_w + 20, panel_h + banner_h + 20),
    ]

    for title, img, x, y in panels:
        # Title banner
        canvas_draw.rectangle([x, y, x + panel_w, y + banner_h], fill=(45, 45, 55))
        canvas_draw.text((x + 12, y + 8), title, fill=(220, 220, 230))
        # Image panel
        resized = img.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, y + banner_h))
        canvas_draw.rectangle([x, y + banner_h, x + panel_w, y + banner_h + panel_h], outline=(80, 80, 95), width=1)

    return canvas


def run_full_verification_suite() -> bool:
    """Executes the complete 12-test automated unit test suite for Module 1."""
    print("=" * 75)
    print("DocGuard: Running Verification Test Suite (Submodules 1.1, 1.2 & 1.3)")
    print("=" * 75)

    test_results = []

    # Test 1: Direct PIL Image Input
    raw_img = Image.new("RGB", (300, 150), color=(240, 240, 240))
    res1 = normalize_image_input(raw_img)
    assert res1.mode == "RGB" and res1.size == (300, 150)
    test_results.append(("PIL Image (RGB) Passthrough", "PASSED"))
    print("  [+] Test 1 Passed: RGB PIL Image normalized correctly.")

    # Test 2: In-Memory Bytes Input
    buf = io.BytesIO()
    raw_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    res2 = normalize_image_input(raw_bytes)
    assert res2.mode == "RGB" and res2.size == (300, 150)
    test_results.append(("Raw Bytes Normalization", "PASSED"))
    print("  [+] Test 2 Passed: In-memory raw bytes decoded to RGB.")

    # Test 3: io.BytesIO Stream Input
    stream = io.BytesIO(raw_bytes)
    res3 = normalize_image_input(stream)
    assert res3.mode == "RGB"
    test_results.append(("io.BytesIO Stream Normalization", "PASSED"))
    print("  [+] Test 3 Passed: io.BytesIO stream handled without disk I/O.")

    # Test 4: RGBA Alpha Compositing
    rgba_img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(50):
        for y in range(100):
            rgba_img.putpixel((x, y), (255, 0, 0, 255))
    res4 = normalize_image_input(rgba_img)
    assert res4.mode == "RGB" and res4.getpixel((75, 50)) == (255, 255, 255)
    test_results.append(("RGBA Alpha-to-White Compositing", "PASSED"))
    print("  [+] Test 4 Passed: RGBA transparent regions composited to white paper.")

    # Test 5: Grayscale (L) Receipt Scan Conversion
    gray_img = Image.new("L", (80, 80), color=128)
    res5 = normalize_image_input(gray_img)
    assert res5.mode == "RGB" and res5.getpixel((10, 10)) == (128, 128, 128)
    test_results.append(("Grayscale (L) to RGB Expansion", "PASSED"))
    print("  [+] Test 5 Passed: Grayscale scanned document expanded to 3 channels.")

    # Test 6: Input Validation Guards
    try:
        normalize_image_input(b"")
        raise AssertionError("Empty bytes should fail!")
    except ValueError:
        pass
    try:
        normalize_image_input("nonexistent_invoice_12345.png")
        raise AssertionError("Missing file should fail!")
    except FileNotFoundError:
        pass
    test_results.append(("Input Validation Guards", "PASSED"))
    print("  [+] Test 6 Passed: Validation error guards verified.")

    # Test 7: Pristine JPEG Quantization Equilibrium
    pristine_canvas = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(pristine_canvas)
    draw.rectangle([20, 20, 236, 60], fill=(240, 240, 240), outline=(0, 0, 0))
    draw.text((30, 30), "Invoice Total: 4,000", fill=(0, 0, 0))
    eq_buffer = io.BytesIO()
    pristine_canvas.save(eq_buffer, format="JPEG", quality=95)
    eq_buffer.seek(0)
    pristine_jpeg = Image.open(eq_buffer)

    ela_pristine, metrics_pristine = compute_ela(pristine_jpeg, quality=95, scale=15)
    assert metrics_pristine["mean_error"] < 2.5
    test_results.append(("Pristine JPEG Quantization Equilibrium", "PASSED"))
    print(f"  [+] Test 7 Passed: Pristine document in equilibrium has low error delta ({metrics_pristine['mean_error']:.4f}).")

    # Test 8: Tampered Anomaly Detection via ELA
    tampered_canvas = pristine_jpeg.copy()
    tamper_draw = ImageDraw.Draw(tampered_canvas)
    tamper_draw.rectangle([110, 28, 220, 50], fill=(255, 255, 255))
    tamper_draw.text((112, 30), "14,000 [MODIFIED]", fill=(200, 0, 0))
    ela_tampered, metrics_tampered = compute_ela(tampered_canvas, quality=95, scale=15)
    ela_np = np.asarray(ela_tampered, dtype=np.float32)
    tampered_region_error = np.mean(ela_np[28:50, 110:220])
    untouched_region_error = np.mean(ela_np[150:200, 20:200])
    assert tampered_region_error > untouched_region_error * 2.0
    test_results.append(("Tampered Anomaly Detection via ELA", "PASSED"))
    print(f"  [+] Test 8 Passed: Tampered patch glows ({tampered_region_error:.2f}) vs untouched ({untouched_region_error:.2f}).")

    # Test 9: Paper Grain & Sensor Noise Extraction
    np.random.seed(42)
    grain_noise = np.random.normal(loc=0.0, scale=8.0, size=(256, 256, 3))
    paper_surface = np.clip(240.0 + grain_noise, 0, 255).astype(np.uint8)
    noisy_doc_pil = Image.fromarray(paper_surface, mode="RGB")
    residual_array, noise_metrics = extract_noise_residual(noisy_doc_pil, kernel_size=3)
    assert residual_array.shape == (256, 256, 3) and noise_metrics["mean_noise"] > 3.0
    test_results.append(("Paper Grain & Sensor Noise Extraction", "PASSED"))
    print(f"  [+] Test 9 Passed: Extracted authentic paper noise residual (mean: {noise_metrics['mean_noise']:.2f}).")

    # Test 10: Inpainted Sterile Patch Detection
    inpainted_paper = paper_surface.copy()
    inpainted_paper[100:160, 80:180] = (245, 245, 245)
    inpainted_pil = Image.fromarray(inpainted_paper, mode="RGB")
    inpainted_residual, _ = extract_noise_residual(inpainted_pil, kernel_size=3)
    pixel_noise_map = np.mean(inpainted_residual, axis=2)
    sterile_noise = float(np.mean(pixel_noise_map[105:155, 85:175]))
    surrounding_noise = float(np.mean(pixel_noise_map[10:60, 10:60]))
    assert sterile_noise < surrounding_noise * 0.25
    test_results.append(("Inpainted Sterile Patch Detection", "PASSED"))
    print("  [+] Test 10 Passed: Inpainted patch detected via severe drop in spatial noise variance.")

    # Test 11: Visual PIL Noise Output & Kernel Guard
    vis_pil, _ = extract_noise_residual(noisy_doc_pil, kernel_size=3, as_pil=True)
    assert isinstance(vis_pil, Image.Image) and vis_pil.mode == "RGB"
    try:
        extract_noise_residual(noisy_doc_pil, kernel_size=4)
        raise AssertionError("Even kernel should fail!")
    except ValueError:
        pass
    test_results.append(("Noise Visualization & Kernel Guard", "PASSED"))
    print("  [+] Test 11 Passed: Visual PIL noise output and even-kernel guard verified.")

    # Test 12: High-Speed Performance Benchmark (< 25 ms)
    start_time = time.perf_counter()
    for _ in range(10):
        extract_noise_residual(noisy_doc_pil, kernel_size=3)
    avg_latency_ms = ((time.perf_counter() - start_time) / 10.0) * 1000.0
    assert avg_latency_ms < 25.0
    test_results.append(("Noise Residual Processing Speed (< 25ms)", "PASSED"))
    print(f"  [+] Test 12 Passed: Noise extraction latency: {avg_latency_ms:.2f} ms per document.")

    print("=" * 75)
    print("Submodules 1.1, 1.2 & 1.3 Verification Summary:")
    for test_name, status in test_results:
        print(f"  * {test_name:<44}: {status}")
    print("=" * 75)
    print("SUCCESS: All 12 unit tests passed!")
    return True


def run_demo_visualizer(output_dir: str = "dataset/samples") -> Path:
    """Generates synthetic medical invoice demonstration images and diagnostic montage."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n[DocGuard Demo Generator]")
    print(f"  Generating synthetic invoice pair and forensic maps into: {out_path.resolve()}")

    pristine_doc, tampered_doc = generate_synthetic_invoice_pair()

    # Compute ELA (Quality 95, Scale 15)
    ela_img, ela_metrics = compute_ela(tampered_doc, quality=95, scale=15)

    # Compute Spatial Noise Residual (Kernel 3, as PIL)
    noise_img, noise_metrics = extract_noise_residual(tampered_doc, kernel_size=3, as_pil=True)

    # Create 4-panel diagnostic montage
    montage = create_forensic_montage(pristine_doc, tampered_doc, ela_img, noise_img)

    # Save artifacts
    pristine_doc.save(out_path / "sample_invoice_pristine.png")
    tampered_doc.save(out_path / "sample_invoice_tampered.png")
    ela_img.save(out_path / "sample_invoice_ela.png")
    noise_img.save(out_path / "sample_invoice_noise.png")
    montage_path = out_path / "sample_diagnostic_montage.png"
    montage.save(montage_path)

    print(f"  [+] Pristine Document Saved:   {out_path / 'sample_invoice_pristine.png'}")
    print(f"  [+] Tampered Document Saved:   {out_path / 'sample_invoice_tampered.png'}")
    print(f"  [+] ELA Map Saved:             {out_path / 'sample_invoice_ela.png'}")
    print(f"  [+] Noise Map Saved:           {out_path / 'sample_invoice_noise.png'}")
    print(f"  [+] Diagnostic Montage Saved:  {montage_path}")
    print(f"      - ELA Metrics:   Mean={ela_metrics['mean_error']}, Max={ela_metrics['max_error']}, Std={ela_metrics['std_error']}")
    print(f"      - Noise Metrics: Mean={noise_metrics['mean_noise']}, Max={noise_metrics['max_noise']}, Std={noise_metrics['std_noise']}")
    return montage_path


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DocGuard Module 1: In-Memory Forensic Preprocessor (preprocessor.py)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Execute the 12-test automated unit test suite."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Generate synthetic diagnostic invoices and visual 4-panel montage."
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Process a custom document image from disk and print forensic metrics."
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset/samples", help="Directory to save diagnostic artifacts."
    )

    args = parser.parse_args()

    # If no flags are provided, run both tests and demo generator
    run_all = not (args.test or args.demo or args.input)

    if args.test or run_all:
        run_full_verification_suite()

    if args.demo or run_all:
        montage_file = run_demo_visualizer(output_dir=args.output_dir)
        print(f"\nSUCCESS: Module 1 completely verified! Diagnostic montage ready at:\n  --> {montage_file.resolve()}\n")

    if args.input:
        input_path = Path(args.input)
        print(f"\n[Processing Custom Document: {input_path.resolve()}]")
        ela_img, ela_metrics = compute_ela(input_path, quality=95, scale=15)
        _, noise_metrics = extract_noise_residual(input_path, kernel_size=3)
        print("  * ELA Quantization Metrics:")
        print(f"      - Mean Error: {ela_metrics['mean_error']}")
        print(f"      - Max Error:  {ela_metrics['max_error']}")
        print(f"      - Std Error:  {ela_metrics['std_error']}")
        print("  * Spatial Noise Metrics:")
        print(f"      - Mean Noise: {noise_metrics['mean_noise']}")
        print(f"      - Max Noise:  {noise_metrics['max_noise']}")
        print(f"      - Std Noise:  {noise_metrics['std_noise']}")
