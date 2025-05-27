from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import shutil
import os
import uuid
from PIL import Image

app = FastAPI()
model = YOLO("runs/detect/custom_YOLOv8FINAL3/weights/best.pt") 

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_filename)

    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            detections.append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": bbox
            })
    os.remove(temp_filename)
    return JSONResponse(content={"detections": detections})
