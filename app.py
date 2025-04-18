import os
import io
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

app = FastAPI(title="YOLOv5 Object Detection API")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

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
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)