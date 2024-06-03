import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class ImageBase64(BaseModel):
    image: str


@app.post("/upload-image/")
async def upload_image(image_data: ImageBase64):
    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data.image)

        # Open the image
        image = Image.open(io.BytesIO(image_bytes))

        model = YOLO('/Users/madebyth/PycharmProjects/cw2-master/cw/runs/detect/train2/weights/best.pt')
        np_image = np.array(image)
        results = model.predict(np_image)
        result = results[0]

        # Convert the image to grayscale
        img_with_prediction = Image.fromarray(result.plot())

        # Save the grayscale image to a bytes buffer
        buffered = io.BytesIO()
        img_with_prediction.save(buffered, format="PNG")
        grayscale_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return JSONResponse(content={"img_with_prediction": grayscale_base64})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)