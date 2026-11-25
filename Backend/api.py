
FastAPI backend for serving the cancer stage and progression prediction model
with optional Grad-CAM heatmap generation.

This API exposes:
- GET /health       : basic health check
- POST /predict     : image upload -> stage + progression prediction (+ heatmap)
"""

import base64
import io
import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from src.dataset import get_transforms
from src.gradcam import GradCAM
from src.model import create_model
from src.utils import load_checkpoint, set_seed


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    os.path.join("models", "best_model.pth"),
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For inference we reuse the validation/test transform (no random augmentation)
INFERENCE_IMG_SIZE = 224
INFERENCE_TRANSFORM = get_transforms(
    phase="val",
    img_size=INFERENCE_IMG_SIZE,
)


def load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Load the trained model from a checkpoint.

    The checkpoint is expected to contain:
    - model_state_dict
    - class_info (with num_stages, num_risk_levels)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = load_checkpoint(model_path)

    class_info = checkpoint.get("class_info")
    if class_info is None:
        # Fallback to defaults if class_info was not stored
        num_stages = 5
        num_risk_levels = 3
    else:
        num_stages = class_info.get("num_stages", 5)
        num_risk_levels = class_info.get("num_risk_levels", 3)

    model = create_model(
        num_stages=num_stages,
        num_risk_levels=num_risk_levels,
        backbone=checkpoint.get("backbone", "resnet50"),
        pretrained=False,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized tensor using the inference transform."""
    return INFERENCE_TRANSFORM(img).unsqueeze(0)


def encode_image_to_base64(img_array: np.ndarray, format: str = "PNG") -> str:
    """
    Encode a NumPy RGB image array as a base64 data URL.
    """
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Cancer Stage Prediction API",
    description=(
        "API for Automatic Cancer Stage and Progression Prediction using "
        "histopathology images with Grad-CAM explainability."
    ),
    version="1.0.0",
)

# CORS so the React frontend (Vite dev server) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model instance (loaded on startup)
MODEL: Optional[torch.nn.Module] = None
DEVICE: str = DEFAULT_DEVICE


@app.on_event("startup")
def startup_event() -> None:
    """
    Load the model once at startup and set seeds for deterministic behavior.
    """
    global MODEL, DEVICE
    set_seed(42)
    try:
        MODEL = load_model(DEFAULT_MODEL_PATH, DEVICE)
    except FileNotFoundError as exc:
        # Defer failure to first prediction request so /health still works
        print(f"[WARN] Could not load model at startup: {exc}")


@app.get("/health")
def health() -> dict:
    """
    Basic health check endpoint.
    """
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": MODEL is not None,
        "model_path": DEFAULT_MODEL_PATH,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(
        default=True,
        description="Whether to generate a Grad-CAM heatmap.",
    ),
    task: str = Query(
        default="stage",
        regex="^(stage|progression)$",
        description="Task for Grad-CAM visualization: 'stage' or 'progression'.",
    ),
) -> dict:
    """
    Predict cancer stage and progression risk from an uploaded histopathology image.

    Returns:
    - predicted stage index and label
    - predicted progression risk index and label
    - probabilities for each class
    - optional Grad-CAM heatmap as base64 PNG
    """
    global MODEL, DEVICE

    if MODEL is None:
        # Try to lazily load the model if startup failed
        try:
            MODEL = load_model(DEFAULT_MODEL_PATH, DEVICE)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Model is not available: {exc}",
            ) from exc

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        ) from exc

    # Preprocess
    input_tensor = pil_to_tensor(image).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(input_tensor)

    # Stage prediction
    stage_logits = outputs["stage_logits"]
    stage_probs = torch.softmax(stage_logits, dim=1)[0]
    stage_idx = int(stage_probs.argmax().item())

    # Progression prediction
    prog_logits = outputs["progression_logits"]
    prog_probs = torch.softmax(prog_logits, dim=1)[0]
    prog_idx = int(prog_probs.argmax().item())

    # Human-readable labels (aligned with config.STAGE_NAMES / RISK_NAMES)
    stage_labels = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    risk_labels = ["Low Risk", "Medium Risk", "High Risk"]

    response: dict = {
        "stage": {
            "index": stage_idx,
            "label": stage_labels[stage_idx] if 0 <= stage_idx < len(stage_labels) else str(
                stage_idx
            ),
            "probabilities": stage_probs.cpu().tolist(),
        },
        "progression": {
            "index": prog_idx,
            "label": risk_labels[prog_idx] if 0 <= prog_idx < len(risk_labels) else str(
                prog_idx
            ),
            "probabilities": prog_probs.cpu().tolist(),
        },
    }

    if explain:
        # Generate Grad-CAM heatmap for the requested task
        gradcam = GradCAM(MODEL)
        heatmap = gradcam.generate_cam(input_tensor, task=task)

        # Overlay on the original RGB image resized to inference size
        rgb_np = np.array(image.resize((INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE)))
        overlaid = gradcam.overlay_heatmap(rgb_np, heatmap, alpha=0.4)

        response["explainability"] = {
            "task": task,
            "heatmap": encode_image_to_base64(overlaid),
        }

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )

"""
FastAPI backend for serving the cancer stage and progression prediction model
with optional Grad-CAM heatmap generation.

This API exposes:
- GET /health       : basic health check
- POST /predict     : image upload -> stage + progression prediction (+ heatmap)
"""

import base64
import io
import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from src.dataset import get_transforms
from src.gradcam import GradCAM
from src.model import create_model
from src.utils import load_checkpoint, set_seed


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    os.path.join("models", "best_model.pth"),
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For inference we reuse the validation/test transform (no random augmentation)
INFERENCE_IMG_SIZE = 224
INFERENCE_TRANSFORM = get_transforms(
    phase="val",
    img_size=INFERENCE_IMG_SIZE,
)


def load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Load the trained model from a checkpoint.

    The checkpoint is expected to contain:
    - model_state_dict
    - class_info (with num_stages, num_risk_levels)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = load_checkpoint(model_path)

    class_info = checkpoint.get("class_info")
    if class_info is None:
        # Fallback to defaults if class_info was not stored
        num_stages = 5
        num_risk_levels = 3
    else:
        num_stages = class_info.get("num_stages", 5)
        num_risk_levels = class_info.get("num_risk_levels", 3)

    model = create_model(
        num_stages=num_stages,
        num_risk_levels=num_risk_levels,
        backbone=checkpoint.get("backbone", "resnet50"),
        pretrained=False,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized tensor using the inference transform."""
    return INFERENCE_TRANSFORM(img).unsqueeze(0)


def encode_image_to_base64(img_array: np.ndarray, format: str = "PNG") -> str:
    """
    Encode a NumPy RGB image array as a base64 data URL.
    """
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Cancer Stage Prediction API",
    description=(
        "API for Automatic Cancer Stage and Progression Prediction using "
        "histopathology images with Grad-CAM explainability."
    ),
    version="1.0.0",
)

# CORS so the React frontend (Vite dev server) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model instance (loaded on startup)
MODEL: Optional[torch.nn.Module] = None
DEVICE: str = DEFAULT_DEVICE


@app.on_event("startup")
def startup_event() -> None:
    """
    Load the model once at startup and set seeds for deterministic behavior.
    """
    global MODEL, DEVICE
    set_seed(42)
    try:
        MODEL = load_model(DEFAULT_MODEL_PATH, DEVICE)
    except FileNotFoundError as exc:
        # Defer failure to first prediction request so /health still works
        print(f"[WARN] Could not load model at startup: {exc}")


@app.get("/health")
def health() -> dict:
    """
    Basic health check endpoint.
    """
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": MODEL is not None,
        "model_path": DEFAULT_MODEL_PATH,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(
        default=True,
        description="Whether to generate a Grad-CAM heatmap.",
    ),
    task: str = Query(
        default="stage",
        regex="^(stage|progression)$",
        description="Task for Grad-CAM visualization: 'stage' or 'progression'.",
    ),
) -> dict:
    """
    Predict cancer stage and progression risk from an uploaded histopathology image.

    Returns:
    - predicted stage index and label
    - predicted progression risk index and label
    - probabilities for each class
    - optional Grad-CAM heatmap as base64 PNG
    """
    global MODEL, DEVICE

    if MODEL is None:
        # Try to lazily load the model if startup failed
        try:
            MODEL = load_model(DEFAULT_MODEL_PATH, DEVICE)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Model is not available: {exc}",
            ) from exc

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        ) from exc

    # Preprocess
    input_tensor = pil_to_tensor(image).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(input_tensor)

    # Stage prediction
    stage_logits = outputs["stage_logits"]
    stage_probs = torch.softmax(stage_logits, dim=1)[0]
    stage_idx = int(stage_probs.argmax().item())

    # Progression prediction
    prog_logits = outputs["progression_logits"]
    prog_probs = torch.softmax(prog_logits, dim=1)[0]
    prog_idx = int(prog_probs.argmax().item())

    # Human-readable labels (aligned with config.STAGE_NAMES / RISK_NAMES)
    stage_labels = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    risk_labels = ["Low Risk", "Medium Risk", "High Risk"]

    response: dict = {
        "stage": {
            "index": stage_idx,
            "label": stage_labels[stage_idx] if 0 <= stage_idx < len(stage_labels) else str(
                stage_idx
            ),
            "probabilities": stage_probs.cpu().tolist(),
        },
        "progression": {
            "index": prog_idx,
            "label": risk_labels[prog_idx] if 0 <= prog_idx < len(risk_labels) else str(
                prog_idx
            ),
            "probabilities": prog_probs.cpu().tolist(),
        },
    }

    if explain:
        # Generate Grad-CAM heatmap for the requested task
        gradcam = GradCAM(MODEL)
        heatmap = gradcam.generate_cam(input_tensor, task=task)

        # Overlay on the original RGB image resized to inference size
        rgb_np = np.array(image.resize((INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE)))
        overlaid = gradcam.overlay_heatmap(rgb_np, heatmap, alpha=0.4)

        response["explainability"] = {
            "task": task,
            "heatmap": encode_image_to_base64(overlaid),
        }

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )

