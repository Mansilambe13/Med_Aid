import streamlit as st
from PyPDF2 import PdfReader
import fitz
import pytesseract
import cv2
import numpy as np
import pandas as pd
import time
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.models import load_model, model_from_json, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from streamlit_extras.add_vertical_space import add_vertical_space
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


from dotenv import load_dotenv
import os
import google.generativeai as genai

# -------------------------
# Load API Key and Setup
# -------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
# gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    return processor, model



# ---------------- PDF Extractor Functions ----------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def recognize_handwritten_text(image_pil):
    processor, model = load_trocr_model()
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def extract_text_from_image_file(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        return recognize_handwritten_text(image)
    except Exception as e:
        return f"Error processing image: {e}"

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        for page_index in range(len(doc)):
            img_list = doc[page_index].get_images(full=True)
            for img in img_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                text += recognize_handwritten_text(image) + "\n"
    except Exception as e:
        text = f"Error extracting text from PDF: {e}"
    return text.strip()


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# def extract_text_from_images(pdf_stream):
#     text = ""
#     try:
#         doc = fitz.open(stream=pdf_stream.getvalue(), filetype="pdf")
#         for page_index in range(len(doc)):
#             img_list = doc[page_index].get_images(full=True)
#             if not img_list:
#                 continue
#             for img_index, img in enumerate(img_list):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image = Image.open(BytesIO(image_bytes)).convert("RGB")
#                 open_cv_image = np.array(image)
#                 processed_image = preprocess_image(open_cv_image)
#                 extracted_text = pytesseract.image_to_string(processed_image, config="--psm 6")
#                 text += extracted_text + "\n"
#     except Exception as e:
#         text = f"Error extracting text from images: {e}"
#     return text.strip()
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
                extracted_text = recognize_handwritten_text(image)
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
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(LABELS), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(r'C:\Users\mansi\OneDrive\Desktop\Documents\Final Year\medaidmodel\pretrained_model.h5')
    return model


def predict_xray_disease(image_file):
    model = load_xray_model()
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)[0]
    result = {LABELS[i]: float(prediction[i]) for i in range(len(LABELS)) if prediction[i] > 0.5}
    return result if result else "No major disease detected with high confidence."

# ---------------- Streamlit App Layout ----------------

st.set_page_config(page_title="MedAidGPT", page_icon="🩺", layout="wide")
st.sidebar.title('🩺 MedAidGPT')
st.sidebar.markdown("### One stop solution for Medical AI Assistance")

tab1, tab2, tab3 = st.tabs(["🏠 Home (GPT Chat)", "📄 PDF Extractor", "🩻 X-ray Analysis"])

# --------- Tab 1: Home with Medical GPT Chat ---------
with tab1:
    st.title("🏠 MedAidGPT")
    # st.markdown("*Welcome to MedAidGPT!*\n\nHere you can:")
    # st.markdown("- 🗎 Extract Medical information from PDFs")
    # st.markdown("- 🩻 Analyze Chest X-rays for disease detection")
    # st.markdown("- 💬 Ask Medical Queries")

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
# --------- Tab 2: PDF Extractor ---------
with tab2:
    st.header("📄 PDF & Image Text Extractor with OCR")
    uploaded_file = st.file_uploader("Upload a handwritten prescription (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])
    extracted_text = ""

    if uploaded_file is not None:
        file_type = uploaded_file.type
        with st.spinner("Extracting text..."):

            if file_type == "application/pdf":
                pdf_stream = BytesIO(uploaded_file.read())
                extracted_text = extract_text_from_pdf(pdf_stream)
                if not extracted_text.strip():
                    st.warning("No selectable text found. Trying OCR...")
                    pdf_stream.seek(0)
                    extracted_text = extract_text_from_pdf(pdf_stream)

            elif file_type.startswith("image/"):
                extracted_text = extract_text_from_image_file(uploaded_file)

            else:
                st.error("Unsupported file type.")

    # Display extracted text if available
    if extracted_text:
        st.subheader("Extracted Text:")
        st.text_area("Extracted Content", extracted_text, height=300)

        # Download extracted text
        text_bytes = extracted_text.encode('utf-8')
        st.download_button(label="📄 Download Extracted Text",
                           data=text_bytes,
                           file_name="extracted_prescription.txt",
                           mime="text/plain")

    # Analyze Prescription button, disabled until text exists
    analyze_disabled = not bool(extracted_text)
    if st.button("Analyze Prescription", disabled=analyze_disabled):
        prompt = ("""
            You are a medical-knowledgeable assistant but not a licensed practitioner. Your task is strictly educational: to help me understand a prescription I’m going to add prescription data at end . You will:

            *List each medication* in the prescription:
            - Generic and brand name (if any)
            - Active ingredient and dose
            - Mechanism of action
            - Common indications
            - Major side effects and contraindications

            *Predict the most likely diagnosis(es)* suggested by this combination of drugs.

            *Outline the available medical treatment plan in context of diagnosis*, based  on the prescription:
            - Duration of that treatment
            - Possible drug–drug or drug–food interactions

            **Suggest home-based supportive treatments remedies measures to take** (e.g., rest, topical remedies, hydration, gentle exercise) that carry minimal risk.

            With proper dietary recommendations** that may support recovery and mitigate side effects, with no prescription-drug interactions.

            *Cite up to two reputable sources* (e.g., UpToDate, WHO, PubMed) for each major point.
            Also you can add additional information to elaborate wherever you deem necessary. Just giving a structure to build but not limiting your response.

            *IMPORTANT*:  
            – This is for informational purposes only and not a substitute for professional medical advice.  
            – I will fact-check everything with authoritative sources and follow up with my doctor.  

            *Now, please analyze the following prescription:*  
        """ + extracted_text)

        with st.spinner('Generating analysis...'):
            analysis_reply = gemini_model.generate_content(prompt).text

        st.subheader("Prescription Analysis:")
        st.write(analysis_reply)

        # Download analysis
        analysis_bytes = analysis_reply.encode('utf-8')
        st.download_button(label="📝 Download Analysis",
                           data=analysis_bytes,
                           file_name="prescription_analysis.txt",
                           mime="text/plain")


# --------- Tab 3: X-ray Disease Prediction ---------
with tab3:
    st.header("🩻 Chest X-Ray Disease Prediction")
    xray_file = st.file_uploader("Upload a Chest X-ray Image", type=['png', 'jpg', 'jpeg'])
    if xray_file is not None:
        st.image(xray_file, "Uploaded X-ray Image", use_column_width=True)
        with st.spinner('Analyzing X-ray...'):
            result = predict_xray_disease(xray_file)
        if isinstance(result, dict):
            st.success("Possible Diagnoses Detected:")
            for disease, score in result.items():
                st.write(f"{disease}** - Confidence: {score:.2f}")
        else:
            st.info(result)

        # New: GPT-based analysis button and output
        if st.button("Deep Dive Analysis with GPT"):
            # Prepare GPT prompt incorporating image context and model output
            context = result if isinstance(result, dict) else {"Result": result}
            prompt = (
                """
                    You are a medically informed assistant, not a licensed practitioner.

                    An X-ray image has been uploaded and analyzed by an AI model. Based on the model's predictions, the following are the most probable pathologies:

                    {context}

                    Analyze this X-ray case using only the image and the predicted pathology data above. Provide a comprehensive explanation of:
                    - What the results suggest clinically
                    - Common causes and consequences of the detected conditions
                    - Any differential diagnosis
                    - Recommended next steps for investigation or monitoring
                    - Non-prescription supportive care suggestions
                                      also you can add aditional inforamtion to elaborate wherever you deemed necessary. just giving a structure to build but not limiting your response.


                    Remember, this is for educational purposes only. Please proceed with the analysis.
                    """
                f"X-ray Findings: {context}\n"
                "(X-ray image displayed above)"
            )
            with st.spinner("Generating GPT analysis..."):
                gpt_reply = gemini_model.generate_content(prompt).text
            st.subheader("GPT Analysis of X-ray:")
            st.write(gpt_reply)