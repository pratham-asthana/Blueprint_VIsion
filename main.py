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

@app.get("/")
def root():
    return {"message": "Blueprint Vision API is live"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        results = model(image)[0]

        detections = []
        for box in results.boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2 - x1, y2 - y1]
            })

        return JSONResponse(content={"detections": detections})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
