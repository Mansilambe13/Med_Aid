import streamlit as st
from PyPDF2 import PdfReader
import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.models import load_model, model_from_json, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from streamlit_extras.add_vertical_space import add_vertical_space


from dotenv import load_dotenv
import os
import google.generativeai as genai

# -------------------------
# Load API Key and Setup
# -------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")

# ---------------- PDF Extractor Functions ----------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_stream):
    text = ""
    try:
        doc = fitz.open(stream=pdf_stream.getvalue(), filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        text = f"Error extracting text: {e}"
    return text.strip()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_images(pdf_stream):
    text = ""
    try:
        doc = fitz.open(stream=pdf_stream.getvalue(), filetype="pdf")
        for page_index in range(len(doc)):
            img_list = doc[page_index].get_images(full=True)
            if not img_list:
                continue
            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                open_cv_image = np.array(image)
                processed_image = preprocess_image(open_cv_image)
                extracted_text = pytesseract.image_to_string(processed_image, config="--psm 6")
                text += extracted_text + "\n"
    except Exception as e:
        text = f"Error extracting text from images: {e}"
    return text.strip()

# ---------------- X-ray Analysis Functions ----------------

LABELS = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
          'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
          'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

@st.cache_resource
def load_xray_model():
    """Load X-ray analysis model and weights"""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(LABELS), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Load trained weights
    model.load_weights(r'C:\Users\mansi\OneDrive\Desktop\Documents\Final Year\medaidmodel\pretrained_model.h5')
    return model

def predict_xray_disease(image_file):
    """Predict disease from X-ray image"""
    model = load_xray_model()
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0]
    result = {LABELS[i]: float(prediction[i]) for i in range(len(LABELS)) if prediction[i] > 0.5}

    if not result:
        return "No major disease detected with high confidence."
    else:
        return result

# ---------------- Streamlit App Layout ----------------

st.set_page_config(page_title="MedAidGPT", page_icon="🩺", layout="wide")
st.sidebar.title('🩺 MedAidGPT')
st.sidebar.markdown("### One stop solution for Medical AI Assistance")

tab1, tab2, tab3 = st.tabs(["🏠 Home (GPT Chat)", "📄 PDF Extractor", "🩻 X-ray Analysis"])

# --------- Tab 1: Home with Medical GPT Chat ---------
with tab1:
    st.title("🏠 MedAidGPT")
    st.markdown("**Welcome to MedAidGPT!**\n\nHere you can:")
    st.markdown("- 🗎 Extract Medical information from PDFs")
    st.markdown("- 🩻 Analyze Chest X-rays for disease detection")
    st.markdown("- 💬 Ask Medical Queries")

    st.markdown("### 💬 Chat with Medical Assistant")

    if "medical_chat_history" not in st.session_state:
        st.session_state.medical_chat_history = []

    medical_input = st.chat_input("Ask any medical question...", key="medical_chat_input")

    if medical_input:
        st.session_state.medical_chat_history.append({"role": "user", "content": medical_input})
        with st.spinner("Analyzing..."):
            med_reply = gemini_model.generate_content(medical_input).text
        st.session_state.medical_chat_history.append({"role": "assistant", "content": med_reply})

    for msg in st.session_state.medical_chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

# --------- Tab 2: PDF Extractor ---------
with tab2:
    st.header("📄 PDF Text Extractor with OCR")
    uploaded_file = st.file_uploader("Upload a Medical PDF", type='pdf')
    
    if uploaded_file is not None:
        pdf_stream = BytesIO(uploaded_file.read())
        extracted_text = extract_text_from_pdf(pdf_stream)

        if not extracted_text.strip():
            st.warning("No selectable text found. Trying OCR...")
            pdf_stream.seek(0)
            extracted_text = extract_text_from_images(pdf_stream)

        if extracted_text.strip():
            st.subheader("Extracted Text:")
            st.text_area("Extracted Content", extracted_text, height=300)
        else:
            st.error("⚠ No text could be extracted. Try another PDF.")

# --------- Tab 3: X-ray Disease Prediction ---------
with tab3:
    st.header("🩻 Chest X-Ray Disease Prediction")
    xray_file = st.file_uploader("Upload a Chest X-ray Image", type=['png', 'jpg', 'jpeg'])

    if xray_file is not None:
        st.image(xray_file, caption="Uploaded X-ray Image", use_column_width=True)
        with st.spinner('Analyzing X-ray...'):
            result = predict_xray_disease(xray_file)

        if isinstance(result, dict):
            st.success("Possible Diagnoses Detected:")
            for disease, score in result.items():
                st.write(f"**{disease}** - Confidence: {score:.2f}")
        else:
            st.info(result)
