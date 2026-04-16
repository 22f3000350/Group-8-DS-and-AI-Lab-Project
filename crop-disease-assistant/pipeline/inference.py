"""
inference.py
------------
Loads the saved MobileNet model and runs prediction on a PIL image.

Your notebook saved two formats:
  - best_mobilenet_full_model.pth  → torch.save(best_model, path)   [full model]
  - best_mobilenet_weights.pkl     → joblib state_dict               [weights only]
  - trial checkpoints              → state['model_state']            [state dict]

This module handles all three, trying full model first (simplest).
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ── Class names (alphabetical = ImageFolder order) ──────────────────────────
# These must match the folder names in your processed/train/ directory exactly.
CLASS_NAMES = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Sugarcane__Bacterial_Blight",
    "Sugarcane__Healthy",
    "Sugarcane__Red_Rot",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Confidence threshold ─────────────────────────────────────────────────────
# Below this, we flag low confidence instead of returning a disease label.
CONFIDENCE_THRESHOLD = 0.60

# ── Eval transform (must match training notebook exactly) ────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model builder (mirrors notebook's build_mobilenet) ───────────────────────
def _build_mobilenet(name: str, dropout: float = 0.2) -> nn.Module:
    """Rebuild the exact architecture used in training."""
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[0] = nn.Dropout(dropout)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:  # mobilenet_v3_large
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[2] = nn.Dropout(dropout)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    return model


def load_model(model_path: str, model_name: str = None, dropout: float = 0.2) -> nn.Module:
    """
    Load the model from disk.

    Parameters
    ----------
    model_path : str
        Path to .pth or .pkl file.
    model_name : str, optional
        'mobilenet_v2' or 'mobilenet_v3_large'.
        Only needed when loading a state_dict (not a full model).
        Check your best_metrics.json → best_params → model to find this.
    dropout : float, optional
        Dropout used during training. Check best_metrics.json → best_params → dropout.

    Returns
    -------
    nn.Module
        Model in eval mode on DEVICE.
    """
    ext = os.path.splitext(model_path)[1].lower()

    # ── Strategy 1: Full model saved with torch.save(model, path) ───────────
    if ext == ".pth":
        try:
            model = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if isinstance(model, nn.Module):
                model.eval()
                print(f"[inference] Loaded full model from {model_path}")
                return model
        except Exception:
            pass  # Fall through to state_dict strategy

        # ── Strategy 2: Checkpoint with state['model_state'] ────────────────
        try:
            state = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if isinstance(state, dict) and "model_state" in state:
                arch = model_name or state.get("model_name", "mobilenet_v2")
                model = _build_mobilenet(arch, dropout)
                model.load_state_dict(state["model_state"])
                model = model.to(DEVICE)
                model.eval()
                print(f"[inference] Loaded state_dict (key=model_state) from {model_path}")
                return model
        except Exception:
            pass

        # ── Strategy 3: Raw state_dict ────────────────────────────────────
        try:
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
            arch = model_name or "mobilenet_v2"
            model = _build_mobilenet(arch, dropout)
            model.load_state_dict(state_dict)
            model = model.to(DEVICE)
            model.eval()
            print(f"[inference] Loaded raw state_dict from {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Could not load model from {model_path}: {e}")

    # ── Strategy 4: joblib .pkl weights ─────────────────────────────────────
    elif ext == ".pkl":
        import joblib
        state_dict = joblib.load(model_path)
        arch = model_name or "mobilenet_v2"
        model = _build_mobilenet(arch, dropout)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        print(f"[inference] Loaded joblib state_dict from {model_path}")
        return model

    else:
        raise ValueError(f"Unsupported model file extension: {ext}")


def predict(model: nn.Module, image: Image.Image) -> dict:
    """
    Run inference on a PIL image.

    Returns
    -------
    dict with keys:
        class_label  : str  e.g. "Corn___Gray_Leaf_Spot"
        crop         : str  e.g. "Corn"
        disease      : str  e.g. "Gray Leaf Spot"
        confidence   : float  e.g. 0.91
        low_confidence: bool  True if confidence < CONFIDENCE_THRESHOLD
        all_probs    : dict  {class_name: probability} for top 5
    """
    # Ensure RGB (handles RGBA, grayscale edge cases)
    image = image.convert("RGB")

    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = torch.topk(probs, 5)
    top5 = {CLASS_NAMES[i]: round(p.item(), 4) for i, p in zip(top5_idx, top5_probs)}

    best_idx   = top5_idx[0].item()
    best_prob  = top5_probs[0].item()
    best_label = CLASS_NAMES[best_idx]

    # Parse crop and disease from label
    # Labels follow two patterns: Crop___Disease or Sugarcane__Disease
    parts  = best_label.replace("__", "___").split("___")
    crop    = parts[0]
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    return {
        "class_label":    best_label,
        "crop":           crop,
        "disease":        disease,
        "confidence":     round(best_prob, 4),
        "low_confidence": best_prob < CONFIDENCE_THRESHOLD,
        "all_probs":      top5,
    }
