from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
async def health_check():
    return JSONResponse(content={"message": "API is running"})


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Minimal stub implementation; replace with model logic as needed.
    return JSONResponse(content={"disease": "unknown", "confidence": 0.0})
