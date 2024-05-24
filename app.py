from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    print(f"Preprocessing image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Sharpen the image
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Resize image to improve OCR accuracy for small text
    processed_image = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return processed_image

def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    pil_image = Image.fromarray(preprocessed_image)
    text = pytesseract.image_to_string(pil_image)
    print(f"Extracted text from image: {text}")
    return text

def convert_pdf_to_images(pdf_path):
    poppler_path = r"C:\poppler-24.02.0\Library\bin"  # Update this path to where your poppler is installed
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    return images

def extract_text_from_pdf(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    all_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        all_text += text
    print(f"Extracted text from PDF: {all_text}")
    return all_text

def extract_text(file_path):
    print(f"Extracting text from file: {file_path}")
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path)
    elif file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file format")

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        extracted_text = extract_text(file_path)
        return f'<h1>Extracted Text:</h1><pre>{extracted_text}</pre>'

if __name__ == "__main__":
    app.run(debug=True)
