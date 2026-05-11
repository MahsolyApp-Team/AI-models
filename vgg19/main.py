from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.models as models
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import io

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

model.load_state_dict(torch.load("vgg19_rgb_model.pth", map_location=device))
model = model.to(device)
model.eval()

import cv2

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
        # Continue to Model
        # =====================
        img_np = np.array(image)

        img_tensor = valid_transform(image=img_np)["image"]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)

        probs = probs.cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_label = idx_to_label[top_idx].replace("___", " ").replace("_", " ")
        confidence = float(probs[top_idx])

        all_probs = {
            idx_to_label[i].replace("___", " ").replace("_", " "): float(p)
            for i, p in enumerate(probs)
        }

        return {
            "prediction": top_label,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }