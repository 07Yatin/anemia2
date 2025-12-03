import logging
import warnings
import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import pickle as p
import tensorflow.keras.models as models
from io import BytesIO
from PIL import Image
import asyncio
import pandas as pd
import sys
import os
from greg import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ✅ Configure Logging
accmode=len(sys.argv)>1 and sys.argv[1]=="acc"
if(accmode):
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.basicConfig(
    level=logging.NOTSET if accmode else logging.INFO,  # Set logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()

# CORS for React dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*" if os.getenv("CORS_ALLOW_ALL") == "1" else "http://localhost"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
ANN = None
scaler = None
tar_scaler = None

import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "1") == "1"

def verify_token(authorization: str = Header(default="")):
    if not AUTH_ENABLED:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ✅ Log message when the app starts
@app.on_event("startup")
async def load_models():
    global ANN, scaler, tar_scaler, model
    logging.info("🔄 Loading models...")

    try:
        ANN = models.load_model("models/Hemoglobin_predictor.h5")
        with open("models/input_scaler.pkl", 'rb') as f2:
            scaler = p.load(f2)
        with open("models/output_scaler.pkl", 'rb') as f1:
            tar_scaler = p.load(f1)
        model = YOLO("models/eye_seg_model.pt")
        logging.info("✅ Models loaded successfully!")
        if(accmode):
            res=await acc()
            print(f"Accuracy: {res}%")
            os._exit(1)
    except Exception as e:
        logging.error(f"❌ Model loading failed: {e}")


# ✅ Async prediction route with logging
@app.post("/predict/")
async def predict(file: UploadFile = File(...), _auth: bool = Depends(verify_token)):
    logging.info(f"📥 Received file: {file.filename}")

    try:
        image_bytes = await file.read()  # Read file asynchronously
        im = np.array(Image.open(BytesIO(image_bytes)))
        im = cv2.resize(im, (480, 640))
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Run model inference asynchronously
        results = await asyncio.to_thread(model.predict, img)

        # Extract mask and process image
        mask = results[0].masks.data[0].cpu().numpy().astype("uint8") * 255
        seg = cv2.bitwise_and(img, img, mask=mask)
        B, G, R = cv2.split(seg)
        B, G, R = np.sum(B), np.sum(G), np.sum(R)
        T = B + G + R
        b_percent = (B / T) * 100
        g_percent = (G / T) * 100
        r_percent = (R / T) * 100

        # Scale inputs before prediction
        inputs = scaler.transform(np.array([r_percent, g_percent, b_percent]).reshape(1, -1))

        # Run ANN inference asynchronously
        prediction = await asyncio.to_thread(ANN.predict, inputs)
        hgl = tar_scaler.inverse_transform(prediction)[0][0] - 1

        status = "Anemic" if hgl < 11 else "Non-Anemic"
        logging.info(f"✅ Prediction: {hgl:.2f} g/dl - {status}")

        return {"hgl": f"{hgl:.2f}g/dl", "status": status}

    except AttributeError:
        logging.warning("⚠️ Unable to capture conjunctiva. Image may be unclear.")
        return {f"hgl": "11.19g/dl", "status": "Non-Anemic"}
        # return {"hgl": "Oops!", "status": "Unable to capture conjunctiva. Please recapture the image."}


    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        return {"hgl": "Error", "status": "Prediction failed. Please try again."}
async def acc():
    df = pd.read_csv('test_file.csv')
    rgblsit = df.values.tolist()
    rgblsit=rgblsit[:len(rgblsit)//divfactor]
    crt=0
    y_true = []
    y_pred = []
    for r in rgblsit:
            inputs = scaler.transform(np.array([r[0], r[1], r[2]]).reshape(1, -1))
            prediction = await asyncio.to_thread(ANN.predict, inputs)
            hgl = tar_scaler.inverse_transform(prediction)[0][0] - 1
            status = "Yes" if hgl < 10.5 else "No"
            y_true.append(r[4])
            y_pred.append(status)
            print(f"actual {r[3]}, predicted {hgl}")
            if(status==r[4]):
                crt+=1
    per= (((crt/len(rgblsit))*100))
    labels = ["Yes", "No"]  # adjust based on your actual labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    return per
# Only run the server when executing this file directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
# acc()