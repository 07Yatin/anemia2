<<<<<<< HEAD
# Anemia Detection Web App  

## Introduction  
The **Anemia Detection Web App** is a powerful and user-friendly tool designed to identify anemia and estimate hemoglobin levels through conjunctiva images. Leveraging advanced computer vision techniques and machine learning models like YOLO for segmentation and ANN for regression, this application provides quick and accurate results. Its primary goal is to assist users in monitoring their health effortlessly and in real-time.

## Features  
- **Conjunctiva Segmentation:** Utilizes YOLO segmentation to extract conjunctiva regions from user-uploaded images.  
- **Hemoglobin Level Estimation:** Employs an ANN model to predict hemoglobin levels based on segmented image data.  
- **Real-time Processing:** Ensures fast and reliable results for user convenience.  
- **Interactive UI:** Built with Gradio for an intuitive and seamless user experience.  
- **Backend Optimization:** FastAPI ensures efficient handling of image inference requests.


 
=======

# Anemia Detection Web App

## Overview
The **Anemia Detection Web App** is a user-friendly tool for detecting anemia and estimating hemoglobin levels from conjunctiva (eye) images. It combines a FastAPI backend for model inference and a Gradio-based frontend for interactive user experience. The app leverages YOLO for image segmentation and an ANN for hemoglobin regression.

## Features
- **Conjunctiva Segmentation:** Uses YOLO to segment the eye region from uploaded images.
- **Hemoglobin Estimation:** Predicts hemoglobin levels using a trained ANN model.
- **Anemia Status:** Classifies as "Anemic" or "Non-Anemic" based on predicted hemoglobin.
- **Interactive Web UI:** Gradio frontend for uploading/capturing images, viewing results, and accessing a chatbot.
- **Diet Recommendations:** Personalized diet tips based on hemoglobin level.
- **User History:** Local storage of flagged results for each user.
- **Anemia Chatbot:** Ask questions about anemia, symptoms, prevention, and more.

## Project Structure
- `main.py` — FastAPI backend for model inference (runs on port 8081)
- `gradioApp.py` — Gradio frontend web app
- `models/` — Contains model weights and scalers
- `user_history/` — Stores user result history
- `Anemia_Dataset.csv` — Dataset (if needed for retraining)

## Setup Instructions
1. **Install dependencies:**
	```powershell
	pip install -r requirements.txt
	```

2. **Ensure model files are present:**
	- `models/eye_seg_model.pt` (YOLO segmentation model)
	- `models/Hemoglobin_predictor.h5` (ANN model)
	- `models/input_scaler.pkl` and `models/output_scaler.pkl` (scalers)

3. **Start the FastAPI backend:**
	```powershell
	python main.py
	```
	This will run the backend server at `http://127.0.0.1:8081`.

4. **Start the Gradio frontend:**
	In a new terminal:
	```powershell
	python gradioApp.py
	```
	This will launch the web app and provide a local URL (and optionally a public share link).

## Usage
1. Open the Gradio web app in your browser (URL shown in terminal).
2. Upload or capture a conjunctiva image.
3. Select your sex and submit.
4. View your predicted hemoglobin, anemia status, and diet recommendations.
5. Optionally, flag/save results with a username for history tracking.
6. Use the chatbot for anemia-related questions.

## Notes
- The app does not store images unless you choose to save results.
- For best results, use clear, well-lit images of the lower eyelid conjunctiva.
- This tool is for educational/screening purposes only. Always consult a clinician for medical advice.

## License
See `LICENSE` for details.
>>>>>>> b73977bfd37622162ca654970b81a517c19625b7
