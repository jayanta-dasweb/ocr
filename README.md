# Flask OCR Application

This is a Flask-based OCR (Optical Character Recognition) application that allows users to upload an image or PDF file, extract text from it, and compare the extracted text with ground truth text to calculate accuracy.

## Features

- Supports image files (`.png`, `.jpg`, `.jpeg`) and PDF files.
- Uses Tesseract OCR to extract text from images.
- Preprocesses images to improve OCR accuracy.
- Allows users to input ground truth text to calculate OCR accuracy.
- Provides an API endpoint for uploading files and getting OCR results in JSON format.

## Prerequisites

- Python 3.x
- Tesseract OCR
- Poppler for PDF processing

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
