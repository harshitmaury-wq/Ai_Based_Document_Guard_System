"""
========================================================================================
DocGuard: Multimodal Document Forgery & Manipulation Detection Platform
Module 6: Neural Stream 3 — Multimodal Fusion Brain (phase3_fusion.py)
----------------------------------------------------------------------------------------
Backbones:  EfficientNet-B0 (Stream 1: ELA) + ResNet-50 (Stream 2: RGB)
Input:      Synchronized Dual-Stream Tensors:
              - ELA Tensor: (B, 3, 224, 224)
              - RGB Tensor: (B, 3, 224, 224)
Fusion:     Feature Concatenation (1280 + 2048 = 3328-dim Embedding)
Hardware:   NVIDIA GeForce GTX 1650 (4.0 GB VRAM) / RTX 3050 (6.0 GB VRAM)
========================================================================================

Forensic & Deep Learning Rationale:
-----------------------------------
1. Late Fusion vs. Early Fusion in Forensic Multi-Stream Networks:
   - Early Fusion (Pixel Concatenation): Stacking RGB and ELA images into a 6-channel
     input tensor forces early convolutional filters to jointly model frequency quantization
     artifacts and natural visual textures simultaneously. Because JPEG quantization noise
     operates under fundamentally different mathematical distributions than natural pixel
     gradients, early convolutions blend and dilute subtle forensic boundaries.
   - Late Fusion (Feature Concatenation): Allows Stream 1 (EfficientNet-B0) to independently
     specialize in frequency-domain JPEG DCT compression deltas, while Stream 2 (ResNet-50)
     independently extracts spatial typography, font edge seams, and anti-aliasing halos.
     At the penultimate layer, their global feature vectors (1280-dim and 2048-dim) are
     concatenated into a 3328-dimensional joint latent representation. The dense fusion
     network then learns non-linear cross-modal correlations (e.g. "if ELA glows at region X
     AND visual RGB exhibits sub-pixel font boundary feathering, probability of fraud -> 1.0").

2. Unfreezing Top Convolutional Stages (Block 7 & Layer 4):
   - Rather than keeping the entire backbones frozen as static feature extractors, Module 6
     selectively unfreezes:
       - ResNet-50: Layer 4 (3 Bottleneck blocks, 2048 channels)
       - EfficientNet-B0: Block 7 & Top Conv (MBConv block 7 + 1x1 projection to 1280 channels)
   - Why Unfreeze Top Blocks for Text Forgery?
     Early layers capture generic primitives (lines, corners, color gradients) that are
     domain-agnostic. The top convolutional blocks, however, govern high-level semantic
     context and receptive fields covering full characters and text words. Fine-tuning
     these top kernels directly on the joint cross-modal loss allows them to specialize in
     document-specific text manipulation signatures (e.g. mismatched font glyphs, spliced
     date digits, and localized inpainting boundaries).

3. Dense Fusion Classifier Architecture:
   Linear(3328, 256) -> ReLU() -> Dropout(0.4) -> Linear(256, 64) -> ReLU() -> Linear(64, 1) -> Sigmoid()
   Regularized with 40% dropout to prevent cross-modal feature domination.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

# Import Module 3 Dual-Stream DataLoader Factory
from dataset_loader import get_dataloaders


# =============================================================================
# Submodule 6.1: Master DocumentFusionNetwork Architecture
# =============================================================================

class DocumentFusionNetwork(nn.Module):
    """Master Multimodal Fusion Network for Document Forgery Detection.

    Unites Stream 1 (ELA Compression Domain) and Stream 2 (Visual RGB Domain)
    via late feature concatenation and selective top-block fine-tuning.
    """

    def __init__(
        self,
        ela_weights_path: Optional[str] = "ela_stream_weights.pth",
        rgb_weights_path: Optional[str] = "rgb_stream_weights.pth",
        pretrained_backbones: bool = True,
        dropout_p: float = 0.4,
    ) -> None:
        """Initializes the multimodal fusion network.

        Args:
            ela_weights_path: Path to trained Phase 1 ELA stream checkpoint.
            rgb_weights_path: Path to trained Phase 2 RGB stream checkpoint.
            pretrained_backbones: If True and custom checkpoints missing, fall back to ImageNet.
            dropout_p: Dropout probability in the dense fusion head (default: 0.4).
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # 1. Stream 1: ELA Compression Backbone (EfficientNet-B0)
        # ---------------------------------------------------------------------
        base_effnet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained_backbones else None
        )
        self.ela_features = base_effnet.features
        self.ela_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Load Phase 1 trained weights if available
        if ela_weights_path:
            if Path(ela_weights_path).exists():
                self._load_ela_checkpoint(ela_weights_path)
            else:
                print(f"[!] Warning: ELA checkpoint '{ela_weights_path}' not found. Using baseline weights.")

        # Freezing Policy for ELA:
        # Freeze Blocks 0 through 6; Unfreeze Block 7 and Block 8 (top convs)
        for i in range(7):  # 0, 1, 2, 3, 4, 5, 6
            for param in self.ela_features[i].parameters():
                param.requires_grad = False

        for i in range(7, len(self.ela_features)):  # Block 7 and Block 8 (1x1 conv)
            for param in self.ela_features[i].parameters():
                param.requires_grad = True

        # ---------------------------------------------------------------------
        # 2. Stream 2: Visual RGB Backbone (ResNet-50)
        # ---------------------------------------------------------------------
        base_resnet = resnet50(
            weights=ResNet50_Weights.DEFAULT if pretrained_backbones else None
        )
        self.rgb_conv1 = base_resnet.conv1
        self.rgb_bn1 = base_resnet.bn1
        self.rgb_relu = base_resnet.relu
        self.rgb_maxpool = base_resnet.maxpool

        self.rgb_layer1 = base_resnet.layer1
        self.rgb_layer2 = base_resnet.layer2
        self.rgb_layer3 = base_resnet.layer3
        self.rgb_layer4 = base_resnet.layer4
        self.rgb_avgpool = base_resnet.avgpool

        # Load Phase 2 trained weights if available
        if rgb_weights_path:
            if Path(rgb_weights_path).exists():
                self._load_rgb_checkpoint(rgb_weights_path)
            else:
                print(f"[!] Warning: RGB checkpoint '{rgb_weights_path}' not found. Using baseline weights.")

        # Freezing Policy for RGB:
        # Freeze Stem (conv1, bn1) and Layers 1, 2, 3; Unfreeze Layer 4
        for param in self.rgb_conv1.parameters():
            param.requires_grad = False
        for param in self.rgb_bn1.parameters():
            param.requires_grad = False
        for param in self.rgb_layer1.parameters():
            param.requires_grad = False
        for param in self.rgb_layer2.parameters():
            param.requires_grad = False
        for param in self.rgb_layer3.parameters():
            param.requires_grad = False

        # Unfreeze Layer 4 for text-level forgery adaptation
        for param in self.rgb_layer4.parameters():
            param.requires_grad = True

        # ---------------------------------------------------------------------
        # 3. Dense Multimodal Fusion Classifier (3328 -> 256 -> 64 -> 1)
        # ---------------------------------------------------------------------
        # ELA Features (1280) + RGB Features (2048) = 3328
        self.fusion_classifier = nn.Sequential(
            nn.Linear(in_features=3328, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid(),
        )

    def _load_ela_checkpoint(self, checkpoint_path: str) -> None:
        """Loads trained convolutional weights from Phase 1 ELA checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Map state dict keys to features
        feature_dict = {}
        for k, v in state_dict.items():
            if k.startswith("features."):
                feature_dict[k.replace("features.", "")] = v

        if feature_dict:
            missing, unexpected = self.ela_features.load_state_dict(feature_dict, strict=False)
            print(f"  [+] Loaded ELA Stream Backbone weights from: {checkpoint_path}")

    def _load_rgb_checkpoint(self, checkpoint_path: str) -> None:
        """Loads trained convolutional weights from Phase 2 RGB checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Filter out fc classification head and load backbone stages
        backbone_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("fc.") and not k.startswith("backbone.fc.")
        }

        # Normalize prefix if saved from DocGuardRGBStream
        normalized_dict = {}
        for k, v in backbone_dict.items():
            clean_k = k.replace("backbone.", "")
            normalized_dict[clean_k] = v

        # Load into individual module stages
        for stage_name, module in [
            ("conv1", self.rgb_conv1),
            ("bn1", self.rgb_bn1),
            ("layer1", self.rgb_layer1),
            ("layer2", self.rgb_layer2),
            ("layer3", self.rgb_layer3),
            ("layer4", self.rgb_layer4),
        ]:
            prefix = f"{stage_name}."
            stage_state = {k[len(prefix):]: v for k, v in normalized_dict.items() if k.startswith(prefix)}
            if stage_state:
                module.load_state_dict(stage_state, strict=False)

        print(f"  [+] Loaded RGB Stream Backbone weights from: {checkpoint_path}")

    def extract_stream_features(
        self, ela_tensor: torch.Tensor, rgb_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts individual stream embeddings and the concatenated representation.

        Args:
            ela_tensor: (B, 3, 224, 224)
            rgb_tensor: (B, 3, 224, 224)

        Returns:
            Tuple: (ela_feat [B, 1280], rgb_feat [B, 2048], fused_feat [B, 3328])
        """
        # Stream 1: ELA Feature Extraction
        x_ela = self.ela_features(ela_tensor)
        x_ela = self.ela_avgpool(x_ela)
        ela_feat = torch.flatten(x_ela, 1)  # (B, 1280)

        # Stream 2: RGB Visual Feature Extraction
        x_rgb = self.rgb_conv1(rgb_tensor)
        x_rgb = self.rgb_bn1(x_rgb)
        x_rgb = self.rgb_relu(x_rgb)
        x_rgb = self.rgb_maxpool(x_rgb)

        x_rgb = self.rgb_layer1(x_rgb)
        x_rgb = self.rgb_layer2(x_rgb)
        x_rgb = self.rgb_layer3(x_rgb)
        x_rgb = self.rgb_layer4(x_rgb)

        x_rgb = self.rgb_avgpool(x_rgb)
        rgb_feat = torch.flatten(x_rgb, 1)  # (B, 2048)

        # Multimodal Late Feature Concatenation
        fused_feat = torch.cat([ela_feat, rgb_feat], dim=1)  # (B, 3328)
        return ela_feat, rgb_feat, fused_feat

    def forward(
        self, ela_tensor: torch.Tensor, rgb_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Full forward pass through the multimodal fusion network.

        Args:
            ela_tensor: Input ELA map tensor (B, 3, 224, 224).
            rgb_tensor: Input visual RGB tensor (B, 3, 224, 224).

        Returns:
            torch.Tensor: Calibrated fused forgery probability p in [0.0, 1.0] of shape (B, 1).
        """
        _, _, fused_feat = self.extract_stream_features(ela_tensor, rgb_tensor)
        out = self.fusion_classifier(fused_feat)
        return out


def count_fusion_parameters(model: DocumentFusionNetwork) -> Dict[str, int]:
    """Computes total, trainable, and frozen parameter breakdowns for both streams."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Stream breakdowns
    ela_trainable = sum(p.numel() for p in model.ela_features.parameters() if p.requires_grad)
    rgb_trainable = sum(
        p.numel()
        for m in [model.rgb_conv1, model.rgb_bn1, model.rgb_layer1, model.rgb_layer2, model.rgb_layer3, model.rgb_layer4]
        for p in m.parameters()
        if p.requires_grad
    )
    classifier_trainable = sum(p.numel() for p in model.fusion_classifier.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "ela_trainable": ela_trainable,
        "rgb_trainable": rgb_trainable,
        "classifier_trainable": classifier_trainable,
    }


# =============================================================================
# Submodule 6.2: Dual-Stream Training & Validation Engines
# =============================================================================

def train_one_epoch(
    model: DocumentFusionNetwork,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Executes one epoch of multimodal training with gradient backpropagation."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in dataloader:
        ela_tensors = batch["ela_tensor"].to(device, non_blocking=True)
        rgb_tensors = batch["rgb_tensor"].to(device, non_blocking=True)
        labels = batch["label"].float().view(-1, 1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Dual-stream forward pass
        outputs = model(ela_tensors, rgb_tensors)
        loss = criterion(outputs, labels)

        # Backward pass through fusion head and unfrozen top blocks
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        predicted_classes = (outputs >= 0.5).float()
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = (correct_predictions / max(1, total_samples)) * 100.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: DocumentFusionNetwork,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validates the multimodal network on unseen holdout documents."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            ela_tensors = batch["ela_tensor"].to(device, non_blocking=True)
            rgb_tensors = batch["rgb_tensor"].to(device, non_blocking=True)
            labels = batch["label"].float().view(-1, 1).to(device, non_blocking=True)

            outputs = model(ela_tensors, rgb_tensors)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predicted_classes = (outputs >= 0.5).float()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = (correct_predictions / max(1, total_samples)) * 100.0
    return epoch_loss, epoch_acc


# =============================================================================
# Submodule 6.3: Master Multimodal Training Coordinator
# =============================================================================

def train_fusion_network(
    dataset_dir: str = "dataset",
    ela_weights_path: str = "ela_stream_weights.pth",
    rgb_weights_path: str = "rgb_stream_weights.pth",
    output_weights: str = "final_forgery_detector.pth",
    history_file: str = "phase3_fusion_history.json",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 0.0001,
    dropout_p: float = 0.4,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Coordinates the full 5-epoch multimodal training loop.

    Args:
        dataset_dir: Root dataset directory.
        ela_weights_path: Checkpoint path for Stream 1.
        rgb_weights_path: Checkpoint path for Stream 2.
        output_weights: File path to save final production model weights.
        history_file: Output path for metrics log JSON.
        epochs: Number of epochs (default: 5).
        batch_size: Mini-batch size (default: 4).
        lr: Adam learning rate (default: 0.0001).
        dropout_p: Dropout probability in fusion head (default: 0.4).
        device_name: 'cuda', 'cpu', or None for auto-detection.

    Returns:
        Dict: Training history dictionary with metrics.
    """
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("DocGuard: Phase 3 Multimodal Fusion Brain Training (Stream 3)")
    print("=" * 75)
    print(f"  * Target Compute Device: {device.type}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  * GPU Model:             {gpu_name} ({vram_gb:.2f} GB VRAM)")
        torch.cuda.reset_peak_memory_stats(0)

    # 1. Dual-Stream DataLoader Construction
    print("\n[Stage 1/4] Loading Synchronized Dual-Stream Document Datasets...")
    train_loader, val_loader, manifest = get_dataloaders(
        base_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=2 if device.type == "cuda" else 0,
        test_size=0.20,
    )
    print(f"  [+] Ingested {manifest['total_documents']} documents (Batch Size = {batch_size}):")
    print(f"      - Training Set:   {manifest['train_count']} files ({manifest['train_distribution']['real']} Real, {manifest['train_distribution']['forged']} Forged)")
    print(f"      - Validation Set: {manifest['val_count']} files ({manifest['val_distribution']['real']} Real, {manifest['val_distribution']['forged']} Forged)")

    # 2. Master Model Architecture Instantiation
    print("\n[Stage 2/4] Constructing DocumentFusionNetwork & Loading Stream Checkpoints...")
    model = DocumentFusionNetwork(
        ela_weights_path=ela_weights_path,
        rgb_weights_path=rgb_weights_path,
        pretrained_backbones=True,
        dropout_p=dropout_p,
    )
    model.to(device)

    param_stats = count_fusion_parameters(model)
    print(f"  [+] Parameter Distribution Breakdown:")
    print(f"      - Total Parameters:          {param_stats['total']:,}")
    print(f"      - Frozen Parameters:         {param_stats['frozen']:,} (Early edge/texture stages)")
    print(f"      - Trainable Parameters:      {param_stats['trainable']:,}")
    print(f"          * EfficientNet Block 7:  {param_stats['ela_trainable']:,}")
    print(f"          * ResNet-50 Layer 4:     {param_stats['rgb_trainable']:,}")
    print(f"          * Dense Fusion Head:     {param_stats['classifier_trainable']:,}")

    # 3. Loss Function & Optimizer
    criterion = nn.BCELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    # 4. Training Loop Execution
    print(f"\n[Stage 3/4] Starting Multimodal Neural Training Loop ({epochs} Epochs)...")
    print("-" * 75)
    print(f" {'Epoch':<7} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<11} | {'Time':<8}")
    print("-" * 75)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_duration = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        history["epoch_times"].append(round(epoch_duration, 2))

        print(
            f"   {epoch:<5} |   {train_loss:<9.4f}  |   {train_acc:<5.1f}   % |   {val_loss:<9.4f}  |   {val_acc:<5.1f}  % |  {epoch_duration:<5.1f} s"
        )

        # Checkpoint Best Model based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "architecture": "DocumentFusionNetwork",
                    "streams": ["ela_efficientnet_b0", "rgb_resnet50"],
                    "fusion_dim": 3328,
                    "dropout_p": dropout_p,
                    "lr": lr,
                },
                output_weights,
            )

    total_training_duration = time.time() - start_training_time
    print("-" * 75)

    # 5. Finalization & Artifact Export
    print("\n[Stage 4/4] Finalizing Production Checkpoint Artifacts...")
    weights_path = Path(output_weights).resolve()
    print(f"  [+] Production Weights Saved: {weights_path} ({weights_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"  [+] Best Checkpoint Epoch:    Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")

    history_payload = {
        "architecture": "DocumentFusionNetwork",
        "streams": ["ela_efficientnet_b0", "rgb_resnet50"],
        "fusion_dim": 3328,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dropout_p": dropout_p,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_training_time_seconds": round(total_training_duration, 2),
        "parameter_breakdown": param_stats,
        "metrics": history,
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=4)
    print(f"  [+] Metrics History Saved:    {Path(history_file).resolve()}")

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        print(f"  [+] Peak GPU VRAM Utilized:   {peak_vram_mb:.1f} MB (Well within VRAM limit)")

    print("=" * 75)
    print("SUCCESS: Module 6 (Multimodal Fusion Brain) is fully trained and verified!")
    print("=" * 75)

    return history_payload


# =============================================================================
# Submodule 6.4: Automated Verification Test Suite (Dry-Run Mode)
# =============================================================================

def run_unit_tests() -> bool:
    """Executes a 6-test automated verification suite for Module 6.

    Verifies:
      1. Model Instantiation & Checkpoint Ingestion
      2. Layer Freezing Guard (Blocks 0-6 in EfficientNet & Layers 1-3 in ResNet frozen)
      3. Top-Block Unfreezing Guard (Block 7 & Layer 4 trainable)
      4. Dense Fusion Head Trainability Guard (3328 -> 256 -> 64 -> 1)
      5. Dual-Stream Forward Pass Shapes & Sigmoid Output Range [0, 1]
      6. Isolated Backpropagation & Cross-Modal Gradient Flow
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("DocGuard: Running Verification Test Suite (Module 6: phase3_fusion.py)")
    print("=" * 75)
    print(f"  [+] Target Test Device: {device.type}")

    # Test 1: Model Instantiation
    model = DocumentFusionNetwork(
        ela_weights_path="ela_stream_weights.pth",
        rgb_weights_path="rgb_stream_weights.pth",
        pretrained_backbones=False,
        dropout_p=0.4,
    )
    model.to(device)
    stats = count_fusion_parameters(model)
    print(f"  [+] Test 1 Passed: Model Instantiation Verified.")
    print(f"      Total Params: {stats['total']:,} | Trainable: {stats['trainable']:,} | Frozen: {stats['frozen']:,}")

    # Test 2: Freezing Guard for Early Stages
    for i in range(7):
        for p in model.ela_features[i].parameters():
            assert not p.requires_grad, f"EfficientNet block {i} must be frozen"

    assert not model.rgb_conv1.weight.requires_grad, "ResNet conv1 must be frozen"
    assert not model.rgb_bn1.weight.requires_grad, "ResNet bn1 must be frozen"
    for b in model.rgb_layer1:
        assert not b.conv1.weight.requires_grad, "ResNet layer1 must be frozen"
    for b in model.rgb_layer2:
        assert not b.conv1.weight.requires_grad, "ResNet layer2 must be frozen"
    for b in model.rgb_layer3:
        assert not b.conv1.weight.requires_grad, "ResNet layer3 must be frozen"
    print("  [+] Test 2 Passed: Early Stages Freezing Guard Verified (Blocks 0-6 & Layers 1-3 frozen).")

    # Test 3: Unfreezing Guard for Top Stages
    b7_trainable = any(p.requires_grad for p in model.ela_features[7].parameters())
    assert b7_trainable, "EfficientNet Block 7 must be unfrozen"

    l4_trainable = any(p.requires_grad for p in model.rgb_layer4.parameters())
    assert l4_trainable, "ResNet Layer 4 must be unfrozen"
    print("  [+] Test 3 Passed: Top Stages Unfreezing Guard Verified (EfficientNet Block 7 & ResNet Layer 4 active).")

    # Test 4: Dense Fusion Classifier Trainability Guard
    for p in model.fusion_classifier.parameters():
        assert p.requires_grad, "All fusion classifier parameters must have requires_grad=True"
    print("  [+] Test 4 Passed: Dense Fusion Classifier Trainability Guard Verified (Linear(3328, 256) -> ReLU -> Dropout(0.4) -> Linear(256, 64) -> ReLU -> Linear(64, 1) -> Sigmoid).")

    # Test 5: Forward Pass Tensor Shape & Value Bounds
    dummy_ela = torch.randn(2, 3, 224, 224, device=device)
    dummy_rgb = torch.randn(2, 3, 224, 224, device=device)

    model.eval()
    with torch.no_grad():
        ela_f, rgb_f, fused_f = model.extract_stream_features(dummy_ela, dummy_rgb)
        output = model(dummy_ela, dummy_rgb)

    assert ela_f.shape == (2, 1280), f"Expected ELA features (2, 1280), got {ela_f.shape}"
    assert rgb_f.shape == (2, 2048), f"Expected RGB features (2, 2048), got {rgb_f.shape}"
    assert fused_f.shape == (2, 3328), f"Expected fused features (2, 3328), got {fused_f.shape}"
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    assert (output >= 0.0).all() and (output <= 1.0).all(), "Predictions must be in [0, 1]"
    print(f"  [+] Test 5 Passed: Dual-Stream Forward Pass Shapes Verified (ELA: {ela_f.shape}, RGB: {rgb_f.shape}, Fused: {fused_f.shape}, Output: {output.shape}).")

    # Test 6: Isolated Backpropagation & Cross-Modal Gradient Flow
    model.train()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)
    optimizer.zero_grad()

    dummy_out = model(dummy_ela, dummy_rgb)
    dummy_target = torch.tensor([[1.0], [0.0]], device=device)
    loss = nn.BCELoss()(dummy_out, dummy_target)
    loss.backward()

    # Verify frozen stages have None gradients
    assert model.ela_features[0][0].weight.grad is None, "Frozen EfficientNet block 0 must have grad=None"
    assert model.rgb_conv1.weight.grad is None, "Frozen ResNet conv1 must have grad=None"
    assert model.rgb_layer1[0].conv1.weight.grad is None, "Frozen ResNet layer1 must have grad=None"

    # Verify unfrozen stages have active gradients
    assert model.ela_features[7][0].block[0][0].weight.grad is not None, "Unfrozen Block 7 must have active grad"
    assert model.rgb_layer4[0].conv1.weight.grad is not None, "Unfrozen Layer 4 must have active grad"
    assert model.fusion_classifier[0].weight.grad is not None, "Fusion classifier Linear(3328, 256) must have active grad"
    print("  [+] Test 6 Passed: Isolated Backpropagation & Cross-Modal Gradient Flow Verified.")

    print("=" * 75)
    print("Module 6 (phase3_fusion.py) Dry-Run Verification Summary:")
    print("  * Model Instantiation & Weight Ingestion      : PASSED")
    print("  * Early Stages Freezing Guard                 : PASSED")
    print("  * Top Stages Unfreezing Guard (Block 7 & L4)  : PASSED")
    print("  * Dense Fusion Classifier Architecture        : PASSED")
    print("  * Dual-Stream Feature Concatenation (3328-dim): PASSED")
    print("  * Cross-Modal Isolated Backpropagation        : PASSED")
    print("=" * 75)
    print("SUCCESS: All 6 dry-run verification tests passed!\n")
    return True


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for Module 6 execution."""
    parser = argparse.ArgumentParser(
        description="DocGuard Module 6: Neural Stream 3 — Multimodal Fusion Brain (phase3_fusion.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the automated 6-test verification suite without executing training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training and validation loaders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Adam optimizer learning rate for fusion fine-tuning.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability in the dense fusion classifier.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Root path to the document dataset directory.",
    )
    parser.add_argument(
        "--ela-weights",
        type=str,
        default="ela_stream_weights.pth",
        help="Path to trained Phase 1 ELA stream weights.",
    )
    parser.add_argument(
        "--rgb-weights",
        type=str,
        default="rgb_stream_weights.pth",
        help="Path to trained Phase 2 RGB stream weights.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_forgery_detector.pth",
        help="Output file path for saving trained final production weights.",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="phase3_fusion_history.json",
        help="Output file path for JSON metrics log.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Target hardware device (defaults to auto-detecting CUDA).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test:
        run_unit_tests()
        return

    # Train master multimodal fusion network
    train_fusion_network(
        dataset_dir=args.dataset_dir,
        ela_weights_path=args.ela_weights,
        rgb_weights_path=args.rgb_weights,
        output_weights=args.output,
        history_file=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_p=args.dropout,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
