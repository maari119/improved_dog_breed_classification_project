# Dog Breed Identifier 🐶

An end‑to‑end **dog breed classifier** web app.  
A FastAPI backend loads a ResNet‑based PyTorch model (trained from the Stanford Dogs dataset) to predict the dog breed from an uploaded image, and a Tailwind‑styled HTML frontend displays the top prediction plus top‑5 alternatives.

---

## Features

- Upload a dog image via drag‑and‑drop or file picker in the browser UI.
- FastAPI backend with robust model loading (supports checkpoints with different `num_classes`).
- ResNet50 (via `timm`) classifier trained on 120 dog breeds (Stanford Dogs) in the notebook.
- Top‑1 and top‑5 predictions with confidence scores in a clean, responsive interface.
- GPU support via PyTorch CUDA wheels (optional but recommended).

---

## Project Structure

```text
.
├── app.py             # FastAPI backend: model loading + /predict API
├── index.html         # Frontend: Tailwind UI + fetch() to backend
├── final-code.ipynb   # Model training / evaluation notebook
├── requirements.txt   # Python dependencies (API + training)
└── README.md          # Project documentation (this file)
```
## Installation
1. Create and activate a virtual environment
```
python -m venv .venv
```
### Linux / macOS
`source .venv/bin/activate`
###  Windows (PowerShell)
`.venv\Scripts\Activate.ps1`
2. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Backend
From the project root, start the FastAPI app with Uvicorn:

```
uvicorn app:app --host 0.0.0.0 --port 8008 --reload
```
--reload watches for code changes and automatically restarts the server (good for development).

The app runs on http://127.0.0.1:8008/ by default
