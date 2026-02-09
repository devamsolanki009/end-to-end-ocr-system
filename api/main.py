from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import pickle
from PIL import Image
import io
from pathlib import Path

from training_2.main_execution import CRNN, ctc_greedy_decode
from preprocessing.image_preprocessing import OCRPreprocessing

# =====================================================
# App initialization
# =====================================================

app = FastAPI(
    title="CRNN OCR API",
    description="FastAPI-based OCR inference using CRNN",
    version="1.0"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Resolve project root safely
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR -> OCR_model/

# =====================================================
# Load label processor (PURE DICT, NOT CLASS)
# =====================================================

LABEL_PATH = BASE_DIR / "label_processor.pkl"
# 👆 If your file is elsewhere, change ONLY this line

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"label_processor.pkl not found at {LABEL_PATH}")

with open(LABEL_PATH, "rb") as f:
    label_processor = pickle.load(f)

alphabet = label_processor["alphabet"]
num_classes = label_processor["num_classes"]

# =====================================================
# Load CRNN model + checkpoint
# =====================================================

model = CRNN(
    img_height=32,
    num_channels=1,
    num_classes=num_classes,
    hidden_size=256
)

CKPT_PATH = BASE_DIR / "checkpoints" / "latest_checkpoint.pth"

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

# =====================================================
# Preprocessing (same as training)
# =====================================================

preprocess = OCRPreprocessing(img_h=32, img_w=128)

# =====================================================
# Routes
# =====================================================

@app.get("/")
def health_check():
    return {
        "status": "OCR API is running",
        "device": str(DEVICE),
        "num_classes": num_classes
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Preprocess
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            prediction = ctc_greedy_decode(outputs, alphabet)[0]

        return JSONResponse(
            content={
                "filename": file.filename,
                "prediction": prediction
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
