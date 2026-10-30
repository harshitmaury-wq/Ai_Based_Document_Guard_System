"""
========================================================================================
DocGuard: Multimodal Document Forgery & Manipulation Detection Platform
Module 8: Inference Engine & XAI Bounding-Box Localizer (predict.py)
----------------------------------------------------------------------------------------
Model:       Master DocumentFusionNetwork (final_forgery_detector.pth)
XAI Engine:  Morphological Localization & Saliency Bounding Box Superimposition
Input:       Document Image (.png, .jpg, .jpeg, .tiff) or Native PDF
Outputs:
  - Global Forgery Risk Score (0.0% to 100.0%)
  - Categorical Forensic Verdict (AUTHENTIC / SUSPICIOUS / FORGED)
  - Annotated Visual Anomaly Map with Bounding Boxes & Translucent Pink Overlay
========================================================================================

Forensic & Explainable AI (XAI) Rationale:
------------------------------------------
1. The Role of XAI in Legal, Regulatory, and Insurance Compliance:
   In high-stakes financial operations, a neural network predicting a black-box probability
   (e.g. "89.4% probability of fraud") is insufficient for legal repudiation or claim denial.
   Human claims adjudicators and judicial courts demand spatial explainability:
     "Which specific figures, names, or stamps were altered?"
   DocGuard bridges neural prediction with Explainable Computer Vision by isolating
   the precise mathematical discrepancy regions and rendering human-interpretable
   bounding boxes directly over the tampered text fields.

2. Morphological Dilation & Contour Grouping Mechanics:
   - Pixel-Level ELA Noise: JPEG quantization differences manifest as disconnected clusters
     of glowing pixels along character strokes and sub-pixel edge seams.
   - Otsu's Dynamic Thresholding: Automatically calculates the bimodal variance threshold
     separating authentic paper equilibrium from anomalous high-frequency energy.
   - Morphological Dilation: Uses a 5x5 rectangular structuring element B to dilate
     isolated character stroke pixels:
         A (+) B = { z | (B_hat)_z INTERSECT A != EMPTY }
     Dilation bridges gaps between individual character glyphs (e.g. '1', '4', ',', '0', '0', '0'),
     coalescing them into a single coherent bounding box around the tampered numerical phrase.
   - Pixel Dust Filtering: Discards spurious camera sensor noise (< 50 px^2) to prevent
     cluttered false alarms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import Core DocGuard Engines
from phase3_fusion import DocumentFusionNetwork
from preprocessor import compute_ela, normalize_image_input


# Standard ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Submodule 8.1: Production Model Loader & Tensor Preprocessor
# =============================================================================

def load_production_model(
    weights_path: Union[str, Path] = "final_forgery_detector.pth",
    device: Optional[torch.device] = None,
) -> DocumentFusionNetwork:
    """Loads the serialized production DocumentFusionNetwork weights.

    Args:
        weights_path: File path to final_forgery_detector.pth.
        device: CUDA or CPU device.

    Returns:
        DocumentFusionNetwork: Model instantiated in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_file = Path(weights_path).resolve()
    if not weights_file.exists():
        raise FileNotFoundError(f"Production weights not found at: {weights_file}")

    model = DocumentFusionNetwork(
        ela_weights_path=None,
        rgb_weights_path=None,
        pretrained_backbones=False,
    )
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_tensors_for_inference(
    rgb_pil: Image.Image,
    ela_pil: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts PIL RGB and ELA images into normalized (1, 3, 224, 224) tensors."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    rgb_tensor = transform(rgb_pil).unsqueeze(0).to(device)
    ela_tensor = transform(ela_pil).unsqueeze(0).to(device)
    return rgb_tensor, ela_tensor


# =============================================================================
# Submodule 8.2: Morphological Anomaly Localizer & Bounding Box Extractor
# =============================================================================

def extract_tamper_bounding_boxes(
    ela_pil: Image.Image,
    min_threshold: int = 40,
    kernel_size: Tuple[int, int] = (5, 5),
    min_area: int = 50,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """Extracts bounding boxes of anomalous regions using Otsu and morphological dilation.

    Args:
        ela_pil: Enhanced Error Level Analysis image from Module 1.
        min_threshold: Minimum brightness threshold for anomaly pixels (default: 40).
        kernel_size: Structuring element dimensions for morphological dilation (default: 5x5).
        min_area: Minimum contour area in pixels to filter out pixel dust (default: 50 px^2).

    Returns:
        Tuple: (List of (x, y, w, h) bounding boxes, binary anomaly mask array)
    """
    # 1. Convert ELA to Grayscale Array
    ela_np = np.array(ela_pil)
    gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)

    # 2. Otsu's Dynamic Bimodal Thresholding
    otsu_thresh_val, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine with conservative minimum threshold guard
    effective_thresh = max(min_threshold, int(otsu_thresh_val))
    _, binary_mask = cv2.threshold(gray, effective_thresh, 255, cv2.THRESH_BINARY)

    # 3. Morphological Dilation: Coalesces individual character strokes into word boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)

    # 4. Find External Contours
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # Add subtle padding (3 pixels) around text
            x_pad = max(0, x - 3)
            y_pad = max(0, y - 3)
            w_pad = min(ela_np.shape[1] - x_pad, w + 6)
            h_pad = min(ela_np.shape[0] - y_pad, h + 6)
            bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))

    # Sort bounding boxes vertically (top-to-bottom reading order)
    bounding_boxes.sort(key=lambda b: (b[1], b[0]))
    return bounding_boxes, dilated_mask


# =============================================================================
# Submodule 8.3: Visual Highlight Rendering & HUD Badge Compositor
# =============================================================================

def render_highlight_overlay(
    original_pil: Image.Image,
    bounding_boxes: List[Tuple[int, int, int, int]],
    risk_score: float,
    verdict: str,
    overlay_opacity: float = 0.25,
) -> Image.Image:
    """Superimposes translucent pink highlights and crisp red bounding boxes.

    Args:
        original_pil: Standardized authentic RGB document image.
        bounding_boxes: List of (x, y, w, h) tamper coordinates.
        risk_score: Global Forgery Risk Score in percent (0-100%).
        verdict: Categorical forensic classification label.
        overlay_opacity: Alpha opacity for pink highlight fills (default: 0.25).

    Returns:
        PIL.Image.Image: Presentation-quality forensic anomaly visualization.
    """
    orig_bgr = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    overlay = orig_bgr.copy()

    # Colors in BGR format
    pink_fill = (180, 105, 255)     # Hot Pink fill for tampered areas
    red_border = (0, 0, 230)        # Crisp Crimson Red for bounding outline
    badge_bg = (30, 30, 30)         # Sleek dark background for header HUD
    badge_green = (50, 180, 50)     # Clean authentic tag
    badge_red = (40, 40, 220)       # Tampered fraud alert tag

    # 1. Draw Translucent Pink Overlays
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), pink_fill, -1)

    # Alpha-blend the pink highlights with the original image
    highlighted = cv2.addWeighted(overlay, overlay_opacity, orig_bgr, 1.0 - overlay_opacity, 0)

    # 2. Draw Crisp Red Bounding Boxes
    for i, (x, y, w, h) in enumerate(bounding_boxes, 1):
        cv2.rectangle(highlighted, (x, y), (x + w, y + h), red_border, thickness=3)

        # Small index label above box
        label_text = f"Tamper #{i}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(th + 4, y - 4)
        cv2.rectangle(highlighted, (x, label_y - th - 3), (x + tw + 6, label_y + 3), red_border, -1)
        cv2.putText(
            highlighted,
            label_text,
            (x + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Render Top Forensic HUD Banner
    h_img, w_img = highlighted.shape[:2]
    hud_height = 42
    hud_canvas = np.zeros((h_img + hud_height, w_img, 3), dtype=np.uint8)
    hud_canvas[:hud_height, :] = badge_bg
    hud_canvas[hud_height:, :] = highlighted

    # HUD Status Badge
    is_tampered = verdict == "FORGED / TAMPERED"
    badge_color = badge_red if is_tampered else badge_green
    status_text = f"DocGuard Forensics | Status: {verdict} | Risk Score: {risk_score:.1f}% | Tamper Count: {len(bounding_boxes)}"

    cv2.putText(
        hud_canvas,
        status_text,
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        badge_color if is_tampered else (100, 255, 100),
        2,
        cv2.LINE_AA,
    )

    return Image.fromarray(cv2.cvtColor(hud_canvas, cv2.COLOR_BGR2RGB))


# =============================================================================
# Submodule 8.4: Master analyze_and_highlight Engine
# =============================================================================

def analyze_and_highlight(
    image_path: Union[str, Path],
    threshold: int = 40,
    output_path: Union[str, Path] = "anomaly_map.png",
    weights_path: str = "final_forgery_detector.pth",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """End-to-end inference and XAI bounding-box localizer for a single document.

    Pipeline:
      1. Normalizes input document and computes in-memory ELA map.
      2. Executes neural inference through DocumentFusionNetwork (0-100% risk score).
      3. Extracts high-variance anomaly regions via Otsu + morphological dilation.
      4. Renders crisp red bounding boxes with translucent pink highlights.
      5. Saves presentation-ready anomaly map to disk.

    Args:
        image_path: Path to the target document file.
        threshold: Minimum brightness threshold for anomaly extraction (default: 40).
        output_path: Path to save the annotated anomaly map image (default: anomaly_map.png).
        weights_path: Production weights checkpoint (default: final_forgery_detector.pth).
        device: CUDA or CPU device.

    Returns:
        Dict: Complete forensic report with global risk score, verdict, and box coordinates.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_file = Path(image_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Target document not found at: {input_file}")

    print("=" * 75)
    print("DocGuard: XAI Inference Engine & Tamper Localizer (predict.py)")
    print("=" * 75)
    print(f"  * Analyzing Document:   {input_file.name}")
    print(f"  * Compute Hardware:     {device.type}")

    # 1. Image Normalization & ELA Generation (Module 1 Core)
    rgb_pil = normalize_image_input(input_file)
    ela_pil, ela_stats = compute_ela(rgb_pil, quality=95, scale=15)
    print(f"  [+] In-memory ELA delta matrix computed (Mean Error: {ela_stats['mean_error']:.4f})")

    # 2. Neural Multimodal Inference (Module 6 Core)
    model = load_production_model(weights_path=weights_path, device=device)
    rgb_tensor, ela_tensor = prepare_tensors_for_inference(rgb_pil, ela_pil, device=device)

    with torch.no_grad():
        tamper_prob = float(model(ela_tensor, rgb_tensor).item())

    risk_score = round(tamper_prob * 100.0, 2)
    verdict = "FORGED / TAMPERED" if risk_score >= 50.0 else "AUTHENTIC / CLEAN"

    print(f"  [+] Global Forgery Risk Score: {risk_score:.2f}%")
    print(f"  [+] Forensic Classification:  {verdict}")

    # 3. Extract Tamper Bounding Boxes (Morphological Localizer)
    bounding_boxes, _ = extract_tamper_bounding_boxes(
        ela_pil=ela_pil,
        min_threshold=threshold,
        kernel_size=(5, 5),
        min_area=50,
    )
    print(f"  [+] Detected {len(bounding_boxes)} suspicious high-frequency anomaly clusters:")
    for i, (x, y, w, h) in enumerate(bounding_boxes, 1):
        print(f"      - Cluster #{i}: BoundingBox(x={x}, y={y}, width={w}, height={h})")

    # 4. Render Highlight Overlay
    annotated_img = render_highlight_overlay(
        original_pil=rgb_pil,
        bounding_boxes=bounding_boxes,
        risk_score=risk_score,
        verdict=verdict,
        overlay_opacity=0.25,
    )

    # 5. Save Output Image
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    annotated_img.save(out_file)
    print(f"  [+] Annotated XAI Anomaly Map saved to: {out_file}")
    print("=" * 75)

    report = {
        "document_name": input_file.name,
        "document_path": str(input_file),
        "global_risk_score": risk_score,
        "forgery_probability": round(tamper_prob, 4),
        "verdict": verdict,
        "is_tampered": bool(verdict == "FORGED / TAMPERED"),
        "tamper_count": len(bounding_boxes),
        "bounding_boxes": [
            {"cluster_id": i, "x": b[0], "y": b[1], "width": b[2], "height": b[3]}
            for i, b in enumerate(bounding_boxes, 1)
        ],
        "ela_statistics": ela_stats,
        "output_visualization_path": str(out_file),
    }
    return report


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DocGuard Module 8: Inference Engine & XAI Bounding-Box Localizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the document image to analyze (.png, .jpg, .jpeg, .tiff).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=40,
        help="Minimum brightness delta threshold for anomaly extraction.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anomaly_map.png",
        help="Path to save the annotated XAI anomaly map.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="final_forgery_detector.pth",
        help="Path to final production model weights checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Hardware execution device (defaults to auto-detecting CUDA).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    analyze_and_highlight(
        image_path=args.image,
        threshold=args.threshold,
        output_path=args.output,
        weights_path=args.weights,
        device=device,
    )


if __name__ == "__main__":
    main()
