import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from backend.services.predict import predict_image
from backend.services.disease_info import disease_info

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Resize for model input (actual normalization is handled in predict_image)
        pil_image = pil_image.resize((224, 224))
        image_np = np.array(pil_image, dtype=np.float32)

        # Model prediction
        result = predict_image(image_np)

        disease_name = result["disease"]
        confidence = result["confidence"]

        # Add knowledge layer
        info = disease_info.get(disease_name, {
            "symptoms": [],
            "treatments": [],
            "prevention": []
        })

        return JSONResponse(content={
            "disease": disease_name,
            "confidence": confidence,
            "symptoms": info["symptoms"],
            "treatments": info["treatments"],
            "prevention": info["prevention"]
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
