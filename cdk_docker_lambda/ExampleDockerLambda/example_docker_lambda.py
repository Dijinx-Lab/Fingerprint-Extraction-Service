from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import asyncio
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

app = FastAPI()
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

semaphore = asyncio.Semaphore(10)

@app.post("/process/extract")
async def api_extract_fingerprints(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided in form data")

    async with semaphore:
        try:
            image_data = await file.read()
            image_stream = BytesIO(image_data)
            data = await process_image_concurrently(image_stream)
            return JSONResponse(content={'success': True, 'data': data, 'message': 'Fingerprint processing completed'}, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

async def process_image_concurrently(file_stream):
    try:
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = resize_image(img, max_height=800, max_width=800)

        output_image_np = await extract_print(img)

        _, processed_image_bytes = cv2.imencode('.png', output_image_np)
        data = {
            'processed_image': processed_image_bytes.tobytes().hex()
        }
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def extract_print(img):

    loop = asyncio.get_event_loop()
   
    output_image = img

    # Convert numpy array to PIL Image and back to numpy array
    output_image_pil = Image.fromarray(output_image)
    output_image_np = np.array(output_image_pil)

    # Enhance image quality
    img_yuv = await loop.run_in_executor(None, cv2.cvtColor, output_image_np, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = await loop.run_in_executor(None, cv2.equalizeHist, img_yuv[:, :, 0])
    img_enhanced = await loop.run_in_executor(None, cv2.cvtColor, img_yuv, cv2.COLOR_YUV2BGR)

    # Additional enhancement using CLAHE for better contrast
    lab = await loop.run_in_executor(None, cv2.cvtColor, img_enhanced, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))#changed from 3.0 to 1.0 and (8,8) to (4,4)
    cl = await loop.run_in_executor(None, clahe.apply, l)
    limg = cv2.merge((cl, a, b))
    img_enhanced = await loop.run_in_executor(None, cv2.cvtColor, limg, cv2.COLOR_Lab2BGR)

    # Convert to HSV for segmentation
    hsv = await loop.run_in_executor(None, cv2.cvtColor, img_enhanced, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 0], dtype="uint8") #changed from 70 to 0
    upper = np.array([20, 255, 255], dtype="uint8")
    mask = await loop.run_in_executor(None, cv2.inRange, hsv, lower, upper)

    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = await loop.run_in_executor(None, cv2.morphologyEx, mask, cv2.MORPH_CLOSE, kernel)
    mask = await loop.run_in_executor(None, cv2.morphologyEx, mask, cv2.MORPH_OPEN, kernel)

    contours, _ = await loop.run_in_executor(None, cv2.findContours, mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]

        # Create a binary mask using the largest contour
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)

        # Perform post-processing
        mask = await loop.run_in_executor(None, cv2.GaussianBlur, mask, (25, 25), 0) #changed from (5,5) to (25,25)

        # Segment the hand or finger region
        hand = cv2.bitwise_and(img_enhanced, img_enhanced, mask=mask)

        # Convert to grayscale and apply adaptive thresholding
        hand_gray = await loop.run_in_executor(None, cv2.cvtColor, hand, cv2.COLOR_BGR2GRAY)
        hand_thresh = await loop.run_in_executor(None, cv2.adaptiveThreshold, hand_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0) #changed from 11,2 to 11,0

        # Apply edge detection
        edges = await loop.run_in_executor(None, cv2.Canny, hand_gray, 100, 200)

        # Find the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image around the bounding box with some padding
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_enhanced.shape[1], x + w + 2 * padding)
        h = min(img_enhanced.shape[0], y + h + 2 * padding)
        cropped_hand = hand_thresh[y:h, x:w]

        return cropped_hand
    else:
        raise HTTPException(status_code=400, detail="No contours found in your image, please retake and upload again")


def resize_image(image, max_height=800, max_width=800):
    height, width = image.shape[:2]
    # Calculate scaling factor if resizing is needed
    if height > max_height or width > max_width:
        scaling_factor = min(max_height / height, max_width / width)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized_image = image
    # Flip the image along the x-axis
    flipped_image = cv2.flip(resized_image, 1)
    return flipped_image