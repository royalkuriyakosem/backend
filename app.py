import os
import io
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import sys
import subprocess

app = FastAPI(title="YOLOv5 Object Detection API")

# Clone YOLOv5 repository if it doesn't exist
if not os.path.exists('yolov5'):
    subprocess.call(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
    
# Add YOLOv5 to path
sys.path.append('./yolov5')

# Load YOLOv5 model
from yolov5.models.common import AutoShape
from yolov5.models.experimental import attempt_load

# Load the model
model_path = 'yolov5s.pt'
if not os.path.exists(model_path):
    torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt', model_path)

model = attempt_load(model_path)
model = AutoShape(model)

@app.get("/")
async def root():
    return {"message": "YOLOv5 Object Detection API is running. Send POST requests to /detect/ endpoint with an image file."}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Perform inference
    results = model(image)
    
    # Process results
    predictions = results.pandas().xyxy[0].to_dict(orient="records")
    
    # Return predictions
    return JSONResponse(content={"detections": predictions})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)