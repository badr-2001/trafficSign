from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

# Load the YOLO model
model = YOLO("best.pt")  # Replace with the path to your YOLO weights file

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Perform detection
        results = model(image)
        detections = []

        # Extract detection results
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names.get(class_id, "Unknown")  # Map class_id to name

                    detections.append({
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "confidence": round(confidence, 4),
                        "class": class_id,
                        "name": class_name,
                    })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
