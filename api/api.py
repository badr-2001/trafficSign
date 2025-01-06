from fastapi import FastAPI, UploadFile, File
import requests
from fastapi.responses import JSONResponse

app = FastAPI()

BACKEND_URL = "http://backend-service.default.svc.cluster.local:8001/predict"

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        file_data = await file.read()
        files = {"file": (file.filename, file_data, file.content_type)}
        
        response = requests.post(BACKEND_URL, files=files)

        return JSONResponse(content=response.json(), status_code=response.status_code)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
