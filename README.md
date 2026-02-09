🧾 End-to-End OCR System using CRNN (with FastAPI Deployment)

This project implements an end-to-end Optical Character Recognition (OCR) system built from scratch using a CRNN (Convolutional Recurrent Neural Network) with CTC loss, trained on synthetic word-level data and deployed as a FastAPI inference service with GPU support.

The project is designed to demonstrate a complete OCR pipeline, including preprocessing, training, inference, and deployment, while also highlighting the limitations of classical OCR models and motivating future extensions using Transformer-based approaches.

🔍 Project Overview

Traditional OCR is not a single task but a pipeline of multiple components.
This project focuses on building and understanding the text recognition stage deeply, and then extending it toward real-world document OCR.

Current Capabilities

Word-level OCR using CRNN + CTC

GPU-accelerated inference

FastAPI-based REST API

Real-time prediction on uploaded images

Supports integration into larger OCR pipelines (e.g., bills, documents)

🧠 OCR Pipeline Design
Phase 1: Text Recognition (Implemented)
Input Image (word-level)
   ↓
Image Preprocessing
   ↓
CRNN (CNN + BiLSTM)
   ↓
CTC Decoding
   ↓
Predicted Text

Phase 2: Document OCR (Planned / In Progress)
Input Image (page / bill)
   ↓
Text Detection (CRAFT / EAST)
   ↓
Crop text regions
   ↓
CRNN Recognition (this model)
   ↓
Reading-order reconstruction
   ↓
Full document text


⚠️ Note: The current CRNN model performs recognition only, not text detection.
Multi-word or document-level OCR requires an additional text detection stage.

🏗️ Model Architecture
CRNN (Convolutional Recurrent Neural Network)

CNN: Extracts visual features from the input image

BiLSTM: Models sequential dependencies across the width of the image

CTC Loss: Enables alignment-free training between image features and text labels

Key Characteristics

Fixed image height (32 px), variable width

Aspect-ratio preserved resizing

Character-level prediction

Greedy CTC decoding

🗂️ Dataset

Dataset: MJSynth (Synthetic Word Dataset)

Type: Word-level, synthetic text images

Why MJSynth:

Large-scale

Clean labels

Ideal for training CRNN-style models

⚙️ Preprocessing Strategy

The preprocessing pipeline is research-grade and OCR-safe:

Grayscale conversion

Fixed height resizing with aspect ratio preservation

Width padding

Pixel normalization

Character-to-index mapping

CTC-compatible label encoding

Custom batch collation for variable-length sequences

To ensure safe deployment, only pure label metadata (alphabet, mappings) is serialized — no custom Python classes are pickled.

🚀 FastAPI Deployment

The trained CRNN model is deployed using FastAPI with the following features:

/ → Health check

/predict → Upload an image and get OCR output

GPU / CPU auto-selection

Swagger UI for interactive testing

Example response:

{
  "filename": "word.png",
  "prediction": "learning"
}

🧪 Observations & Limitations
Expected Behavior

Works well on single words

Performs reliably on clean printed text

Minor CTC-related errors (e.g., duplicated characters) are expected

Known Limitations

Cannot read full pages directly

No layout understanding

Requires text detection for document OCR

Greedy CTC decoding may produce repeated characters

These limitations are intentional and form the basis for future improvements.

🔮 Future Work

Planned extensions include:

✅ Text detection integration (CRAFT / EAST)

🔄 Document-level OCR pipeline (bills, receipts)

🔬 Transformer-based OCR (TrOCR / VisionEncoderDecoder)

📊 CRNN vs Transformer comparison (CER, WER, latency)

🐳 Dockerized deployment

🌐 Frontend UI for live OCR demos

📁 Project Structure
OCR_model/
├── api/                  # FastAPI application
├── preprocessing/        # Image and label preprocessing
├── training/             # CRNN training pipeline
├── checkpoints/          # Model checkpoints (.pth)
├── text_detection/       # (Planned) Text detection module
├── utils/                # Helper functions
├── README.md

🧠 Key Learnings

OCR is a multi-stage problem, not just a model

CRNN + CTC is powerful but limited in layout understanding

Deployment requires careful handling of serialization

Transformers simplify OCR but come with trade-offs

Classical OCR pipelines are still highly relevant