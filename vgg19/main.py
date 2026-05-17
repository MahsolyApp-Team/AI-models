from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.models as models
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import io
import os
import httpx
import cv2
from dotenv import load_dotenv

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Label Map
# =====================
label_map = {
    'Tomato___Late_blight' : 0,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus' : 1,
    'Peach___healthy' : 2,
    'Grape___Esca_(Black_Measles)' : 3,
    'Potato___Late_blight' : 4,
    'Pepper,_bell___Bacterial_spot' : 5,
    'Strawberry___healthy' : 6,
    'Orange___Haunglongbing_(Citrus_greening)' : 7,
    'Tomato___Leaf_Mold' : 8,
    'Apple___Black_rot' : 9,
    'Strawberry___Leaf_scorch' : 10,
    'Tomato___Early_blight' : 11,
    'Cherry_(including_sour)___healthy' : 12,
    'Corn_(maize)___Common_rust_' : 13,
    'Blueberry___healthy' : 14,
    'Potato___Early_blight' : 15,
    'Pepper,_bell___healthy' : 16,
    'Apple___Cedar_apple_rust' : 17,
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)' : 18,
    'Tomato___Tomato_mosaic_virus' : 19,
    'Tomato___Target_Spot' : 20,
    'Tomato___healthy' : 21,
    'Peach___Bacterial_spot' : 22,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot' : 23,
    'Tomato___Spider_mites Two-spotted_spider_mite' : 24,
    'Corn_(maize)___healthy' : 25,
    'Squash___Powdery_mildew' : 26,
    'Cherry_(including_sour)___Powdery_mildew' : 27,
    'Tomato___Bacterial_spot' : 28,
    'Grape___Black_rot' : 29,
    'Apple___healthy' : 30,
    'Potato___healthy' : 31,
    'Corn_(maize)___Northern_Leaf_Blight' : 32,
    'Tomato___Septoria_leaf_spot' : 33,
    'Raspberry___healthy' : 34,
    'Soybean___healthy' : 35,
    'Apple___Apple_scab' : 36,
    'Grape___healthy' : 37
}

idx_to_label = {v: k for k, v in label_map.items()}

# =====================
# Transform
# =====================
MEAN = [0.4664, 0.4891, 0.4104]
STD = [0.1993, 0.1751, 0.2176]

valid_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

# =====================
# Load Model
# =====================
model = models.vgg19(weights=None)

for param in model.features.parameters():
    param.requires_grad = False

num_in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_in_features, len(label_map))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "vgg19_rgb_model.pth"), map_location=device))

model = model.to(device)
model.eval()

# =====================
# Leaf Detection
# =====================
LEAF_GREEN_THRESHOLD = 0.08
LEAF_YELLOW_THRESHOLD = 0.08

def detect_leaf(image: Image.Image):
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]

    img_float = img_array.astype(np.float32) / 255.0
    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]

    # Green detection
    green_mask = (g > r + 0.05) & (g > b + 0.05) & (g > 0.15)
    green_ratio = green_mask.mean()

    # Yellow detection
    yellow_mask = (r > 0.3) & (g > 0.3) & (b < 0.35) & (np.abs(r - g) < 0.3)
    yellow_ratio = yellow_mask.mean()

    # Edge complexity
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_complexity = np.sum(edges > 0) / (h * w)

    is_leaf = (green_ratio + yellow_ratio) / 2 > LEAF_GREEN_THRESHOLD or green_ratio > 0.05

    return is_leaf, {
        "green_ratio": float(green_ratio),
        "yellow_ratio": float(yellow_ratio),
        "edge_complexity": float(edge_complexity)
    }

# =====================
# GrabCut Segmentation
# =====================

# Foreground ratio thresholds — outside this range the mask is considered invalid
GRABCUT_MIN_FOREGROUND_RATIO = 0.10  # all-black mask → GrabCut saw nothing
GRABCUT_MAX_FOREGROUND_RATIO = 0.97  # all-white mask → GrabCut couldn't separate fg/bg


def _is_mask_valid(mask_binary: np.ndarray) -> bool:
    """Return True only if the mask has a meaningful foreground region."""
    foreground_ratio = mask_binary.mean()
    return GRABCUT_MIN_FOREGROUND_RATIO < foreground_ratio < GRABCUT_MAX_FOREGROUND_RATIO


def segment_leaf(image: Image.Image) -> np.ndarray:
    """
    Apply GrabCut segmentation to isolate the leaf and replace the background
    with neutral gray (128, 128, 128) — close to the ImageNet pixel mean —
    so the model focuses on the leaf rather than background clutter.

    If GrabCut produces an invalid mask (all-black, all-white, or raises an
    exception), the original image is returned unchanged so inference is not
    harmed by a bad segmentation.

    Returns a uint8 RGB numpy array ready for the albumentations pipeline.
    """
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # ---- Run GrabCut ----
    try:
        mask = np.zeros(img_bgr.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Rect covers most of the image, leaving a small border for background samples
        rect = (10, 10, w - 20, h - 20)
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Definite/probable foreground → 1, everything else → 0
        mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    except Exception as e:
        print(f"[WARN] GrabCut exception: {e} — using original image.")
        return img_rgb

    # ---- Quality gate: fall back if mask is unusable ----
    if not _is_mask_valid(mask_binary):
        fg_ratio = float(mask_binary.mean())
        print(f"[WARN] GrabCut mask rejected (fg_ratio={fg_ratio:.3f}) — using original image.")
        return img_rgb

    # ---- Good mask: blend leaf with neutral gray background ----
    # Smooth hard edges to avoid border artifacts
    mask_blurred = cv2.GaussianBlur(mask_binary * 255, (15, 15), 0)

    # Float alpha [0, 1] with extra channel dim for broadcasting
    mask_alpha = (mask_blurred / 255.0)[:, :, np.newaxis]

    # Neutral gray background — close to ImageNet pixel mean
    gray_bg = np.full(img_rgb.shape, 128, dtype=np.uint8)

    # Alpha-blend: leaf pixels from original, background pixels from gray
    segmented = (img_rgb * mask_alpha + gray_bg * (1.0 - mask_alpha)).astype(np.uint8)
    return segmented


# =====================
# Gemini Treatment Plan
# =====================

# Ordered list of Gemini models to try (fallback chain)
GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_PROMPT_TEMPLATE = """You are a plant pathology expert. For the plant disease "{disease_name}", provide a brief structured response with exactly these 3 sections:

Symptoms: (2-3 key visible symptoms)
Treatment: (2-3 actionable treatment steps)
Prevention: (2-3 prevention tips)

Be concise. No extra text."""


async def get_treatment_plan(disease_name: str) -> dict:
    """
    Try each Gemini model in order. Return treatment plan on first success.
    If all models fail, return a service unavailable notice.
    """
    if not GEMINI_API_KEY:
        return {
            "treatment_plan": None,
            "treatment_plan_status": "Treatment plan service is not available now."
        }

    prompt = GEMINI_PROMPT_TEMPLATE.format(disease_name=disease_name)
    last_error = ""

    async with httpx.AsyncClient(timeout=15.0) as client:
        for model_name in GEMINI_MODELS:
            print(f"Trying Gemini model: {model_name}")
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={GEMINI_API_KEY}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 300,
                    "temperature": 0.3
                }
            }

            try:
                response = await client.post(url, json=payload)

                # Quota / rate-limit errors → try next model
                if response.status_code in (429, 503, 500):
                    last_error = f"{model_name}: HTTP {response.status_code}"
                    continue

                if response.status_code != 200:
                    last_error = f"{model_name}: HTTP {response.status_code}"
                    continue

                data = response.json()

                # Extract text safely
                candidates = data.get("candidates", [])
                if not candidates:
                    last_error = f"{model_name}: empty candidates"
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    last_error = f"{model_name}: empty parts"
                    continue

                treatment_text = parts[0].get("text", "").strip()
                if not treatment_text:
                    last_error = f"{model_name}: empty text"
                    continue

                # Success — parse into structured sections
                return {
                    "treatment_plan": parse_treatment_sections(treatment_text),
                    "treatment_plan_status": "ok",
                    "model_used": model_name
                }

            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = f"{model_name}: {str(e)}"
                continue

    # All models exhausted
    return {
        "treatment_plan": None,
        "treatment_plan_status": "Treatment plan service is not available now."
    }


def parse_treatment_sections(text: str) -> dict:
    """Parse Gemini's structured response into a clean dict."""
    sections = {"symptoms": "", "treatment": "", "prevention": ""}
    current_key = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("symptoms:"):
            current_key = "symptoms"
            sections[current_key] = line.split(":", 1)[-1].strip()
        elif lower.startswith("treatment:"):
            current_key = "treatment"
            sections[current_key] = line.split(":", 1)[-1].strip()
        elif lower.startswith("prevention:"):
            current_key = "prevention"
            sections[current_key] = line.split(":", 1)[-1].strip()
        elif current_key:
            # Continuation lines
            sections[current_key] += " " + line

    return sections


# =====================
# FastAPI App
# =====================
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Plant Disease API"}

# =====================
# Prediction Endpoint
# =====================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # =====================
        # Leaf Detection
        # =====================
        is_leaf, details = detect_leaf(image)

        if not is_leaf:
            return {
                "message": "No leaf detected. Please upload a clear plant leaf image.",
                "tips": [
                    "Make sure the leaf is clearly visible",
                    "Avoid complex backgrounds",
                    "Ensure good lighting"
                ]
            }

        # =====================
        # GrabCut Segmentation
        # Isolate the leaf and replace the background with neutral gray (128, 128, 128)
        # before passing to the model. Falls back to the original image automatically
        # if the mask is invalid (all-black, all-white, or GrabCut raises an exception).
        # =====================
        img_np = segment_leaf(image)

        # =====================
        # Model Inference
        # =====================
        img_tensor = valid_transform(image=img_np)["image"]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)

        probs = probs.cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_label = idx_to_label[top_idx].replace("___", " ").replace("_", " ")
        confidence = float(probs[top_idx])

        # =====================
        # Gemini Treatment Plan
        # =====================
        gemini_result = await get_treatment_plan(top_label)

        response = {
            "prediction": top_label,
            "confidence": confidence,
        }

        if gemini_result["treatment_plan"] is not None:
            response["treatment_plan"] = gemini_result["treatment_plan"]
            response["model_used"] = gemini_result.get("model_used", "unknown")
        else:
            response["treatment_plan_status"] = gemini_result["treatment_plan_status"]

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }