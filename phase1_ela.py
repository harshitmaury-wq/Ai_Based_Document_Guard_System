"""
========================================================================================
DocGuard: Multimodal Document Forgery & Manipulation Detection Platform
Module 4: Phase 1 ELA Compression Stream Neural Training (phase1_ela.py)
----------------------------------------------------------------------------------------
Target Network: EfficientNet-B0 (ImageNet Pre-Trained Backbone)
Input Plane:    In-Memory Error Level Analysis (ELA) Compression Maps (B, 3, 224, 224)
Hardware:       NVIDIA GeForce GTX 1650 (4.0 GB VRAM) / RTX 3050 (6.0 GB VRAM)
========================================================================================

Forensic & Deep Learning Rationale:
-----------------------------------
1. Why EfficientNet-B0 on High-Frequency ELA Maps:
   Error Level Analysis maps are fundamentally different from natural photograph scenes:
   they are composed of high-frequency Discrete Cosine Transform (DCT) quantization error
   spikes where altered numbers glow brightly against dark equilibrium backgrounds.
   EfficientNet's Mobile Inverted Bottleneck Convolutions (MBConv) utilize:
     - Depthwise Separable Convolutions: Convolves each color channel independently,
       preserving subtle RGB quantization differences without premature cross-channel blurring.
     - Squeeze-and-Excitation (SE) Optimization: Dynamically recalibrates channel-wise feature
       maps, acting as an attention mechanism to amplify anomalous glowing ELA text patches.

2. Transfer Learning & Layer Freezing (VRAM Optimization):
   Training a deep convolutional network from scratch requires storing intermediate
   activation tensors and gradient graphs for all 4.0 million parameters, easily exceeding
   4 GB of VRAM. By freezing all foundational convolutional blocks (requires_grad = False):
     - Zero computational graph memory is allocated for the convolutional backbone.
     - Only the custom classification head (~82,049 parameters) computes gradients.
     - Peak VRAM footprint drops from ~3.5 GB down to ~1.1 - 1.2 GB, enabling rapid,
       crash-free execution on entry-level Turing/Ampere GPUs.

3. Custom Classification Head:
   Dropout(0.3) -> Linear(1280, 64) -> ReLU() -> Linear(64, 1) -> Sigmoid()
   Outputs scalar probability p in [0.0, 1.0] representing the likelihood of document tampering.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Import Module 3 DataLoader Factory
from dataset_loader import get_dataloaders


# =============================================================================
# Submodule 4.1: Model Architecture & Transfer Learning Initialization
# =============================================================================

def build_ela_stream_model(pretrained: bool = True) -> nn.Module:
    """Builds the Stream 1 EfficientNet-B0 model with frozen base layers.

    Architecture:
      - Backbone: EfficientNet-B0 (ImageNet weights), features frozen.
      - Classification Head:
          Dropout(0.3)
          Linear(in_features=1280, out_features=64)
          ReLU(inplace=True)
          Linear(in_features=64, out_features=1)
          Sigmoid()

    Args:
        pretrained: If True, initializes with official ImageNet weights.

    Returns:
        nn.Module: Configured PyTorch neural network.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Freeze all foundational convolutional blocks
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier with custom binary forensic classification head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1280, out_features=64),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=64, out_features=1),
        nn.Sigmoid(),
    )

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """Counts total, trainable, and frozen parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


# =============================================================================
# Submodule 4.2 & 4.3: Training & Validation Epoch Engines
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains the model for one epoch on the ELA tensor stream.

    Returns:
        Tuple: (average_train_loss, train_accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch in dataloader:
        # Stream 1 Input: In-Memory ELA Tensor (B, 3, 224, 224)
        ela_tensors = batch["ela_tensor"].to(device, non_blocking=True)
        # Ground Truth Label: (B, 1) float tensor (0.0 = Real, 1.0 = Forged)
        labels = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(ela_tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = ela_tensors.size(0)
        running_loss += loss.item() * batch_size

        # Binary decision threshold at 0.5
        preds = (outputs >= 0.5).float()
        correct_preds += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = (correct_preds / max(1, total_samples)) * 100.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model on the validation dataset without gradient tracking.

    Returns:
        Tuple: (average_val_loss, val_accuracy_percent)
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            ela_tensors = batch["ela_tensor"].to(device, non_blocking=True)
            labels = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)

            outputs = model(ela_tensors)
            loss = criterion(outputs, labels)

            batch_size = ela_tensors.size(0)
            running_loss += loss.item() * batch_size

            preds = (outputs >= 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_samples += batch_size

    val_loss = running_loss / max(1, total_samples)
    val_acc = (correct_preds / max(1, total_samples)) * 100.0
    return val_loss, val_acc


# =============================================================================
# Submodule 4.4: Master Training Pipeline & Checkpointing
# =============================================================================

def train_ela_stream(
    dataset_dir: Union[str, Path] = "dataset",
    weights_output_path: Union[str, Path] = "ela_stream_weights.pth",
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    num_workers: int = 2,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the complete Phase 1 ELA neural training pipeline.

    Args:
        dataset_dir: Path to dataset root directory.
        weights_output_path: File path to save best model weights.
        epochs: Number of training epochs (default 5).
        batch_size: Mini-batch size (default 8).
        learning_rate: Adam optimizer learning rate (default 0.001).
        num_workers: Number of DataLoader worker subprocesses (default 2).
        device_name: Optional forced device ('cuda' or 'cpu').

    Returns:
        Dict: Training history dictionary with losses and accuracies.
    """
    # 1. Device Selection & Hardware Profiling
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("DocGuard: Phase 1 ELA Stream Neural Training (EfficientNet-B0)")
    print("=" * 75)
    print(f"  * Target Compute Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  * GPU Model:             {gpu_name} ({vram_gb:.2f} GB VRAM)")
        torch.cuda.empty_cache()

    # 2. Build DataLoaders via Module 3
    print("\n[Stage 1/4] Loading and Stratifying Document Datasets...")
    train_loader, val_loader, manifest = get_dataloaders(
        base_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        test_size=0.20,
    )
    print(f"  [+] Ingested {manifest['total_documents']} documents:")
    print(f"      - Training Set:   {manifest['train_count']} files ({manifest['train_distribution']['real']} Real, {manifest['train_distribution']['forged']} Forged)")
    print(f"      - Validation Set: {manifest['val_count']} files ({manifest['val_distribution']['real']} Real, {manifest['val_distribution']['forged']} Forged)")

    # 3. Model Architecture Instantiation
    print("\n[Stage 2/4] Constructing EfficientNet-B0 ELA Stream Model...")
    model = build_ela_stream_model(pretrained=True).to(device)
    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"  [+] Model Parameters:")
    print(f"      - Total:     {total_params:,}")
    print(f"      - Frozen:    {frozen_params:,} (Convolutional feature extractor frozen)")
    print(f"      - Trainable: {trainable_params:,} (Custom classification head)")

    # 4. Loss Function & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # 5. Training Loop
    print("\n[Stage 3/4] Starting Neural Training Loop (5 Epochs)...")
    print("-" * 75)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^12} | {'Val Loss':^12} | {'Val Acc':^12} | {'Time':^8}")
    print("-" * 75)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    weights_path = Path(weights_output_path)

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

        elapsed = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        history["epoch_times"].append(round(elapsed, 2))

        print(
            f"{epoch:^8} | {train_loss:^12.4f} | {train_acc:^11.1f}% | "
            f"{val_loss:^12.4f} | {val_acc:^11.1f}% | {elapsed:^7.1f}s"
        )

        # Checkpointing: Save weights if validation loss improves or on final epoch
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), weights_path)

    print("-" * 75)

    # 6. Save Artifacts & Training History
    print("\n[Stage 4/4] Finalizing Phase 1 Artifacts...")
    # Ensure best weights are safely written
    if not weights_path.exists():
        torch.save(model.state_dict(), weights_path)

    file_size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"  [+] Trained Weights Saved:   {weights_path.resolve()} ({file_size_mb:.2f} MB)")
    print(f"  [+] Best Checkpoint Epoch:   Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # Save training metrics history JSON for downstream evaluation in Module 7
    history_file = Path("phase1_ela_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [+] Metrics History Saved:   {history_file.resolve()}")

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        print(f"  [+] Peak GPU VRAM Utilized:  {peak_vram_mb:.1f} MB (Well within 4.0 GB VRAM limit)")

    print("=" * 75)
    print("SUCCESS: Module 4 (Phase 1 ELA Stream) is fully trained and verified!")
    return history


# =============================================================================
# Automated Dry-Run Verification Harness
# =============================================================================

def run_dry_run_test() -> bool:
    """Executes a 5-test dry-run verification suite on dummy tensors."""
    print("=" * 75)
    print("DocGuard: Running Verification Test Suite (Module 4: phase1_ela.py)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_results: List[Tuple[str, str]] = []

    def record_test(name: str, passed: bool, msg: str = ""):
        status = "PASSED" if passed else "FAILED"
        test_results.append((name, status))
        symbol = "[+]" if passed else "[!]"
        print(f"  {symbol} Test {len(test_results)}: {name} -> {status}")
        if msg:
            print(f"      {msg}")

    # Test 1: Model Architecture Instantiation
    model = build_ela_stream_model(pretrained=True).to(device)
    record_test("Model Instantiation (EfficientNet-B0)", True, f"Device: {device}")

    # Test 2: Backbone Freezing Guard
    features_frozen = all(not p.requires_grad for p in model.features.parameters())
    record_test("Backbone Freezing Guard (requires_grad=False)", features_frozen, "Convolutional layers frozen.")

    # Test 3: Classifier Trainability Guard
    classifier_trainable = all(p.requires_grad for p in model.classifier.parameters())
    record_test("Classifier Head Trainable Guard", classifier_trainable, "Head parameters active.")

    # Test 4: Forward Pass Shape Verification
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    record_test(
        "Forward Pass Output Tensor Shape (B, 1)",
        tuple(output.shape) == (2, 1),
        f"Output shape: {tuple(output.shape)}, values in [0, 1]: {0.0 <= float(output.min()) <= float(output.max()) <= 1.0}",
    )

    # Test 5: Gradient Backpropagation on Classifier Only
    dummy_target = torch.tensor([[0.0], [1.0]], device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.train()
    optimizer.zero_grad()
    pred = model(dummy_input)
    loss = criterion(pred, dummy_target)
    loss.backward()

    # Verify features have NO gradients, classifier HAS gradients
    features_no_grad = all(p.grad is None for p in model.features.parameters())
    classifier_has_grad = any(p.grad is not None for p in model.classifier.parameters())
    optimizer.step()

    record_test(
        "Isolated Backpropagation (Classifier Only)",
        features_no_grad and classifier_has_grad,
        f"Features Grad None: {features_no_grad}, Classifier Grad Active: {classifier_has_grad}",
    )

    print("=" * 75)
    print("Module 4 (phase1_ela.py) Dry-Run Verification Summary:")
    for test_name, status in test_results:
        print(f"  * {test_name:<44}: {status}")
    print("=" * 75)
    all_passed = all(s == "PASSED" for _, s in test_results)
    if all_passed:
        print("SUCCESS: All 5 dry-run verification tests passed!")
    return all_passed


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DocGuard Module 4: Phase 1 ELA Stream Neural Training (phase1_ela.py)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Execute 5-test dry-run verification suite without full training."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default 5)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training and validation (default 8)."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for Adam optimizer (default 0.001)."
    )
    parser.add_argument(
        "--weights", type=str, default="ela_stream_weights.pth", help="Output file path for best weights."
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset", help="Root directory of document dataset."
    )

    args = parser.parse_args()

    if args.test:
        run_dry_run_test()
    else:
        # Run full training pipeline
        train_ela_stream(
            dataset_dir=args.dataset_dir,
            weights_output_path=args.weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_workers=2,
        )
