from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import asyncio
from io import BytesIO
import cv2
import numpy as np
# from rembg import remove
# from PIL import Image

app = FastAPI()
handler = Mangum(app)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

semaphore = asyncio.Semaphore(10)

@app.post("/process/extract")
async def api_extract_fingerprints(file: UploadFile = File(...), rmbg: bool = Query(False)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided in form data")

    async with semaphore:
        try:
            image_data = await file.read()
            image_stream = BytesIO(image_data)
            data = await process_image_concurrently(image_stream, rmbg, False)
            return JSONResponse(content={'success': True, 'data': data, 'message': 'Fingerprint processing completed'}, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# @app.post("/process/remove-bg")
# async def api_remove_background(file: UploadFile = File(...)):
#     if not file:
#         raise HTTPException(status_code=400, detail="No file provided in form data")

#     async with semaphore:
#         try:
#             image_data = await file.read()
#             image_stream = BytesIO(image_data)
#             data = await process_image_concurrently(image_stream, False, True)
#             return JSONResponse(content={'success': True, 'data': data, 'message': 'Background removal completed'}, status_code=200)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

async def process_image_concurrently(file_stream, remove_bg, remove_only_request):
    try:
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = resize_image(img, max_height=800, max_width=800)

        # if remove_only_request:
        #     output_image_np = await remove_background(img)
        # else:
        #     output_image_np = await extract_print(img, remove_bg)
        # REMOVE IF BG REMOVAL IS REQUIRED
        output_image_np = await extract_print(img, remove_bg)

        _, processed_image_bytes = cv2.imencode('.png', output_image_np)
        data = {
            'processed_image': processed_image_bytes.tobytes().hex()
        }
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# async def remove_background(img):
#     loop = asyncio.get_event_loop()
#     output_image = await loop.run_in_executor(None, remove, img)
#     output_image_pil = Image.fromarray(output_image)
#     output_image_np = np.array(output_image_pil)
#     return output_image_np

async def extract_print(img, remove_bg):
    # if remove_bg:
    #     img = await remove_background(img)

    loop = asyncio.get_event_loop()
    hsv = await loop.run_in_executor(None, cv2.cvtColor, img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    mask = await loop.run_in_executor(None, cv2.inRange, hsv, lower, upper)

    contours, _ = await loop.run_in_executor(None, cv2.findContours, mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        hand = cv2.bitwise_and(img, img, mask=mask)
        hand_gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        hand_thresh = cv2.adaptiveThreshold(hand_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1], x + w + 2 * padding)
        h = min(img.shape[0], y + h + 2 * padding)
        cropped_hand = hand_thresh[y:h, x:w]
        return cropped_hand
    else:
        raise HTTPException(status_code=400, detail="No contours found!")

def resize_image(image, max_height=800, max_width=800):
    height, width = image.shape[:2]
    if height > max_height or width > max_width:
        scaling_factor = min(max_height / height, max_width / width)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image
    return image