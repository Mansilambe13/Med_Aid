MedAidGPT is a Streamlit-based Medical AI application that combines LLM-powered medical chat, PDF prescription analysis with OCR, and AI-based Chest X-ray disease detection into a single unified interface.


⚠️ Disclaimer: This project is intended strictly for educational and research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.

✨ Features
💬 Medical GPT Chat

Conversational medical assistant powered by Google Gemini

Handles general medical queries

Maintains session-based chat history

Designed for explanatory and educational responses

📄 Medical PDF & Prescription Analyzer

Upload medical PDFs (typed or scanned)

Dual extraction pipeline:

Direct text extraction (PyMuPDF)

OCR-based extraction (Tesseract + OpenCV)

One-click Prescription Analysis using Gemini:

Medication breakdown (dose, mechanism, side effects)

Possible diagnosis inference

Treatment overview

Drug–drug and drug–food interactions

Dietary and home-care recommendations

References to reputed medical sources

🩻 Chest X-ray Disease Detection

Upload chest X-ray images (.jpg, .png)

AI model based on DenseNet121

Multi-label disease detection (14 pathologies), including:

Pneumonia

Cardiomegaly

Effusion

Pneumothorax

Fibrosis

Edema

Confidence-based filtering for meaningful predictions

Optional GPT-based clinical interpretation of model output

🧠 AI & ML Stack
Component	Technology
LLM	Google Gemini (gemini-2.0-flash-lite)
X-ray Model	DenseNet121 (Transfer Learning)
OCR	Tesseract OCR + OpenCV
PDF Parsing	PyMuPDF (fitz), PyPDF2
Frontend	Streamlit
Image Processing	OpenCV, PIL
Deep Learning	TensorFlow / Keras




The images are for testing purposes.

