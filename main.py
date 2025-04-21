from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from pathlib import Path
import shutil
import os
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import uvicorn
from config import T
import io



# FastAPI app initialization
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
loaded_object_name = None
loaded_reference_features = None
loaded_reference_surroundings = []

# Configure Google Gemini AI
GENAI_API_KEY = "AIzaSyDym4CkIvpXxY5kGIKnoHxF1pQI6HuZUqg"
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Load Pakistani currency detection model
currency_model = load_model("pakistani_currency_model_final.h5")

# Currency label mapping
label_map = {0: '10 PKR', 1: '20 PKR', 2: '50 PKR', 3: '100 PKR', 4: '500 PKR', 5: '1000 PKR', 6: '5000 PKR'}

# Load YOLOv8 models
main_yolo_model = YOLO("yolov8l.pt")  # Main YOLOv8 model
custom_yolo_model = YOLO("best.pt")   # Fine-tuned YOLOv8 model

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import shutil
from pathlib import Path

app = FastAPI()

# Globals
loaded_object_name = None
loaded_reference_descriptors = None
loaded_reference_keypoints = None

# ---------------------------
# Helper Functions
# ---------------------------

def ensure_grayscale(image):
    """Convert to grayscale if image is not already grayscale."""
    if image is None:
        raise ValueError("Image could not be loaded.")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def extract_orb_features(image, nfeatures=2000):
    """Extract ORB features from the image."""
    gray = ensure_grayscale(image)
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, edgeThreshold=15, fastThreshold=5)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def filter_good_matches(reference_descriptors, query_descriptors):
    """Filter good matches between reference and query descriptors."""
    if reference_descriptors is None or query_descriptors is None:
        return []

    # FLANN parameters for ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(reference_descriptors, query_descriptors, k=2)
        # If FLANN returns empty results
        if not matches:
            return []
    except cv2.error as e:
        print(f"FLANN match error: {e}")
        return []

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

def calculate_similarity_score(good_matches, reference_keypoints, query_keypoints):
    """Calculate the similarity score based on good matches."""
    if not good_matches:
        return 0.0

    min_keypoints = min(len(reference_keypoints), len(query_keypoints))
    if min_keypoints == 0:
        return 0.0

    score = len(good_matches) / min_keypoints
    return min(score * 2.0, 1.0)  # Boost a bit, max 1.0

# ---------------------------
# API Endpoints
# ---------------------------

@app.post("/capture_object/")
async def capture_object(file: UploadFile = File(...), object_name: str = Form(...)):
    try:
        base_dir = Path("object_data") / object_name
        base_dir.mkdir(parents=True, exist_ok=True)

        temp_video_path = Path("temp_videos") / f"{object_name}.mp4"
        Path("temp_videos").mkdir(exist_ok=True)

        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract frames from video
        cap = cv2.VideoCapture(str(temp_video_path))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 20 == 0:  # Save every 20th frame
                frame_path = base_dir / f"{object_name}_{frame_count}.jpg"
                cv2.imwrite(str(frame_path), frame)

        cap.release()
        return JSONResponse(content={"message": f"Captured object frames for '{object_name}'"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/load_object/")
async def load_object(object_name: str = Form(...)):
    global loaded_object_name, loaded_reference_descriptors, loaded_reference_keypoints

    object_dir = Path("object_data") / object_name
    if not object_dir.exists():
        return JSONResponse(status_code=404, content={"detail": "Object not found"})

    all_descriptors = []
    all_keypoints = []

    for image_path in object_dir.glob("*.jpg"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue  # Skip if the image cannot be loaded
        keypoints, descriptors = extract_orb_features(image)

        if descriptors is not None:
            all_descriptors.append(descriptors)
            all_keypoints.extend(keypoints)

    if not all_descriptors:
        return JSONResponse(status_code=400, content={"detail": "No features found in object images"})

    loaded_object_name = object_name
    loaded_reference_descriptors = np.vstack(all_descriptors)
    loaded_reference_keypoints = all_keypoints

    return JSONResponse(content={"message": f"Loaded object '{object_name}' successfully"})

@app.post("/search_object/")
async def search_object(file: UploadFile = File(...)):
    global loaded_object_name, loaded_reference_descriptors, loaded_reference_keypoints

    if loaded_reference_descriptors is None or loaded_object_name is None:
        return JSONResponse(status_code=400, content={"detail": "No object loaded. Please load an object first."})

    try:
        # Log the uploaded file's name
        print(f"Uploaded file name: {file.filename}")

        # Check file extension (case-insensitive)
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
        if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
            print(f"File extension not supported: {file.filename}")
            return JSONResponse(status_code=400, content={"detail": "Unsupported file format. Please upload a valid image file."})

        # Read query image
        query_image_bytes = await file.read()
        
        if len(query_image_bytes) == 0:
            return JSONResponse(status_code=400, content={"detail": "Uploaded file is empty."})

        np_arr = np.frombuffer(query_image_bytes, np.uint8)
        query_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Debugging: Check if image is loaded
        if query_image is None:
            return JSONResponse(status_code=400, content={"detail": "Failed to load the query image. Check the file format."})

        # Extract features from query image
        query_keypoints, query_descriptors = extract_orb_features(query_image)

        if query_descriptors is None:
            return JSONResponse(status_code=400, content={"detail": "No descriptors found in the uploaded image"})

        # Match features
        good_matches = filter_good_matches(loaded_reference_descriptors, query_descriptors)

        # Calculate similarity
        score = calculate_similarity_score(good_matches, loaded_reference_keypoints, query_keypoints)

        response = {
            "object_name": loaded_object_name,
            "total_query_keypoints": len(query_keypoints),
            "good_matches": len(good_matches),
            "similarity_score": round(score, 3)
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Error: {str(e)}"})

# Generate caption route
@app.post("/generate_caption")
async def generate_caption(request: Request):
    data = await request.json()
    image_base64 = data.get("image_base64")

    if not image_base64:
        return {"error": "No image provided"}

    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    response = model.generate_content([T, {"mime_type": "image/jpeg", "data": img_bytes}])

    return {"caption": response.text}

# Detect objects route
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        file_content_type = file.content_type
        file_extension = file.filename.split('.')[-1].lower()

        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(content={"error": "Invalid or unsupported image format"}, status_code=400)

        image_np = np.array(image)

        results_main = main_yolo_model(image_np, conf=0.5)
        results_custom = custom_yolo_model(image_np, conf=0.5)

        detected_objects = []

        def process_results(results, model_name):
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence >= 0.3:
                        class_id = int(box.cls[0])
                        class_name = model_name.names[class_id]

                        if class_name == "2":
                            class_name = "door"

                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        detected_objects.append({
                            "class": class_name,
                            "confidence": round(confidence, 2),
                            "bbox": [x_min, y_min, x_max, y_max]
                        })

        process_results(results_main, main_yolo_model)
        process_results(results_custom, custom_yolo_model)

        return JSONResponse(content={
            "detected_objects": detected_objects,
            "file_type": file_content_type,
            "file_extension": file_extension
        }, status_code=200)

    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

# Predict currency route
@app.post("/predict_currency")
async def predict_currency(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img = preprocess_currency_image(image)

        predictions = currency_model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return JSONResponse(content={
            "currency": label_map[predicted_class],
            "confidence": confidence
        }, status_code=200)

    except Exception as e:
        print(f"Error processing currency detection: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

# Detect emotion route
@app.post("/detect_emotion")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()

        # Prepare image for Gemini API (no cropping, full image)
        prompt = "Give one word emotion from angry, disgust, fear, happy, neutral, sad, surprise. If there is no face in the image just return 'face not found'"

        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])

        emotion = response.text.strip().lower()

        return {"emotion": emotion}

    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return {"error": "Internal server error"}

# Function to preprocess image for currency classification
def preprocess_currency_image(image: Image.Image):
    img_height, img_width = 128, 128
    img = np.array(image)
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12345)
