import io
import json
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

Model = YOLO('FastAPI-Object-Detection-using-pre-trained-yolov8-model-Upload-Image/AI_Model/your_model.pt')

app = FastAPI()


async def detection(model, img_content, confidence):
    with Image.open(img_content) as img:
        
        # img = Image.open(img_content)

        result = model(img, device='cpu', conf=confidence)
        detection = {}
        data = json.loads(result[0].tojson()) 


        if len(data) == 0:
            res = {"AI": "No Detection"}
            detection.update(res)

        else:
            for item in data:
                obj_name = item['name']
                conf = item['confidence']
                box = item['box']
                res = {obj_name: {'Confidence': conf, 'Box': box}}
                detection.update(res)

        # img.close()

    return detection


@app.get("/status")
async def status():
    return "AI Server is running"


@app.post("/face")
async def create_items(file: UploadFile = File(...)):
    image = io.BytesIO(await file.read())
    try:
        results = await detection(Model, image, 0.95)
        return results
    except Exception as e:
        return {"AI": f"Error: {str(e)}"}


# pip install -r requirements.txt
# python -m uvicorn Code.main:app --reload
# pip install -upgrade ultralytics
