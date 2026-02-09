import torch
import pickle
from PIL import Image
from preprocessing.image_preprocessing import OCRPreprocessing
from training_2.main_execution import CRNN, ctc_greedy_decode
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_label_processor(path="preprocessing_data.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model(checkpoint_path, label_processor):
    model = CRNN(
        img_height=32,
        num_channels=1,
        num_classes=label_processor["num_classes"],
        hidden_size=256
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model

def preprocess_image(image: Image.Image):
    transform = OCRPreprocessing(img_h=32, img_w=128)
    image = image.convert("L")
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image