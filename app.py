from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from src.config import WEIGHTS_PATH, DATA_PATH, DEVICE
from src.yolo_detector import YOLODetector
from keras._tf_keras.keras.models import load_model
app = FastAPI()

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount the output directory to make images accessible via URL
app.mount("/output", StaticFiles(directory="output"), name="output")

# Initialize the YOLO detector
detector = YOLODetector(
    weights_path=WEIGHTS_PATH,
    data_path=DATA_PATH,
    device=DEVICE
)

class ResNetClassifier:
    """A class to perform image classification using a pre-trained ResNet model.

    This classifier is specifically designed to classify head and neck cancer endoscopy images
    into referral and non-referral categories.

    Attributes:
        model: Loaded Keras model for classification
        height (int): Required height for input images
        width (int): Required width for input images
        class_names (list): List of class names ['non-referral', 'referral']
    """
    def __init__(self, model_path, input_shape=(224, 224)):
        """Initialize the ResNet classifier.

        Args:
            model_path (str): Path to the saved Keras model file
            input_shape (tuple): Tuple of (height, width) for input images. Defaults to (224, 224)
        """
        self.model = load_model(model_path)
        self.height, self.width = input_shape
        self.class_names = ['non-referral', 'referral']
        
    def preprocess_image(self, img):
        """Preprocess an image for inference.

        Performs color conversion, resizing, normalization, and adds batch dimension.

        Args:
            img (numpy.ndarray): Input image in BGR format (OpenCV default)

        Returns:
            numpy.ndarray: Preprocessed image ready for model inference
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input shape
        img = cv2.resize(img, (self.width, self.height))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    def predict(self, img):
        """Perform classification on an input image.

        Args:
            img (numpy.ndarray): Input image in BGR format

        Returns:
            dict: Dictionary containing:
                - predicted_label (str): 'referral' or 'non-referral'
                - confidence (float): Confidence score for the prediction
                - class_probabilities (dict): Probabilities for each class
        """
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img, verbose=0)[0]
        
        # Extract the scalar value from the prediction array
        pred_value = prediction.item()  # This gets a single scalar value
        
        # Now use the scalar value for comparisons and conversions
        predicted_class = int(pred_value >= 0.5)
        confidence = float(pred_value) if predicted_class == 1 else float(1 - pred_value)
        
        return {
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': {
                'non-referral': float(1 - pred_value),
                'referral': float(pred_value)
            }
        }

# Initialize ResNet classifier
resnet_classifier = ResNetClassifier("models/sankeerthana_resnet50_model.hdf5")


@app.get("/")
async def root():
    """Root endpoint to check API status.

    Returns:
        dict: Status message and operational status
    """
    return {"message": "Welcome to the Head and Neck Cancer Glottis Detection API", "status": "operational"}

@app.post("/image-detect-glottis")
async def image_detect_glottis(file: UploadFile = File(...)):
    """Process a single image to detect glottis using YOLO.

    Args:
        file (UploadFile): Uploaded image file

    Returns:
        JSONResponse: Contains:
            - status: Success/error status
            - image_url: URL to access the annotated image
            - detections: List of detected objects with bounding boxes

    Raises:
        HTTPException: If image processing fails or invalid input
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
        
        # Process image
        result = detector.process_image(image)
        
        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        filepath = OUTPUT_DIR / filename
        
        # Save the annotated image
        cv2.imwrite(str(filepath), result["annotated_image"])
        
        # Construct the URL for the saved image
        image_url = f"/output/{filename}"
        
        # Prepare detection results
        detection_results = []
        for det in result["detections"]:
            detection_results.append({
                "label": det["class"],
                "confidence": round(det["confidence"], 3),
                "bbox": {
                    "x1": det["bbox"][0],
                    "y1": det["bbox"][1],
                    "x2": det["bbox"][2],
                    "y2": det["bbox"][3]
                }
            })
        
        # Return structured response
        return JSONResponse(content={
            "status": "success",
            "image_url": image_url,
            "detections": detection_results
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/video-detect-glottis")
async def video_detect_glottis(file: UploadFile = File(...)):
    """Process a video to detect glottis and classify frames.

    Performs both YOLO detection for glottis and ResNet classification on qualifying frames.
    If no qualifying frames are found, falls back to sampling frames at regular intervals.

    Args:
        file (UploadFile): Uploaded video file

    Returns:
        JSONResponse: Contains:
            - status: Success/error status
            - message: Processing status message
            - video_url: URL to access the processed video
            - overall_classification: Overall classification results
            - qualifying_frames/sampled_frames: Information about processed frames
            - video_info: Video metadata (dimensions, fps, etc.)

    Raises:
        HTTPException: If video processing fails or invalid input
    """
    try:
        # Save uploaded video temporarily
        temp_path = OUTPUT_DIR / f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # Open video file
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            os.remove(temp_path)
            return JSONResponse(
                status_code=400,
                content={"error": "Could not open video file"}
            )

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate minimum bbox requirements
        min_bbox_width = frame_width * 0.2
        min_bbox_height = frame_height * 0.3
        min_aspect_ratio = 1.5

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_video_{timestamp}.mp4"
        output_path = OUTPUT_DIR / output_filename
        frames_folder = OUTPUT_DIR / f"qualifying_frames_{timestamp}"
        frames_folder.mkdir(exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        frame_count = 0
        qualifying_frames = 0
        qualifying_frames_info = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with YOLO
            result = detector.process_image(frame)
            out.write(result["annotated_image"])
            
            # Check for qualifying frames
            for det in result["detections"]:
                bbox = det["bbox"]
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                aspect_ratio = bbox_height/bbox_width

                # Check if frame meets qualifying criteria
                if (det["confidence"] > 0.8 and 
                    bbox_width >= min_bbox_width and 
                    bbox_height >= min_bbox_height and
                    bbox_height > bbox_width and
                    aspect_ratio >= min_aspect_ratio):
                    
                    if qualifying_frames < 10:
                        # Save qualifying frame with full annotation
                        frame_filename = f"frame_{qualifying_frames:02d}.jpg"
                        frame_path = frames_folder / frame_filename
                        cv2.imwrite(str(frame_path), result["annotated_image"])
                        
                        # Perform ResNet classification on the original frame
                        resnet_result = resnet_classifier.predict(frame)
                        
                        # Store frame info with both YOLO and ResNet results
                        qualifying_frames_info.append({
                            "frame_number": frame_count,
                            "url": f"/output/qualifying_frames_{timestamp}/{frame_filename}",
                            "yolo_detection": {
                                "confidence": det["confidence"],
                                "bbox": {
                                    "x1": bbox[0],
                                    "y1": bbox[1],
                                    "x2": bbox[2],
                                    "y2": bbox[3],
                                    "width": bbox_width,
                                    "height": bbox_height,
                                    "aspect_ratio": aspect_ratio
                                }
                            },
                            "resnet_classification": resnet_result,
                            "frame_dimensions": {
                                "width": frame_width,
                                "height": frame_height
                            }
                        })
                        qualifying_frames += 1
                    break

            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")

        # Release resources
        cap.release()
        out.release()
        os.remove(temp_path)

        # Handle case where no qualifying frames were found
        if not qualifying_frames_info:
            print("No qualifying frames found. Attempting to process all frames...")
            
            # Reopen the video to process all frames
            cap = cv2.VideoCapture(str(temp_path))
            sampled_frames_info = []
            frame_count = 0
            
            # Sample every 30th frame for analysis
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % 30 == 0:  # Process every 30th frame
                    try:
                        # Perform ResNet classification on sampled frame
                        resnet_result = resnet_classifier.predict(frame)
                        
                        # Save sampled frame
                        frame_filename = f"sampled_frame_{frame_count:04d}.jpg"
                        frame_path = frames_folder / frame_filename
                        cv2.imwrite(str(frame_path), frame)
                        
                        sampled_frames_info.append({
                            "frame_number": frame_count,
                            "url": f"/output/qualifying_frames_{timestamp}/{frame_filename}",
                            "resnet_classification": resnet_result,
                            "frame_dimensions": {
                                "width": frame_width,
                                "height": frame_height
                            }
                        })
                        
                        if len(sampled_frames_info) >= 10:  # Limit to 10 sampled frames
                            break
                            
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {str(e)}")
                        
                frame_count += 1
                
            cap.release()
            
            # Calculate classification based on sampled frames
            if sampled_frames_info:
                referral_count = sum(1 for frame in sampled_frames_info 
                                   if frame["resnet_classification"]["predicted_label"] == "referral")
                                   
                overall_classification = {
                    "predicted_label": "referral" if referral_count >= len(sampled_frames_info)/2 else "non-referral",
                    "confidence_score": referral_count / len(sampled_frames_info),
                    "referral_frame_count": referral_count,
                    "total_sampled_frames": len(sampled_frames_info),
                    "note": "Classification based on sampled frames due to no qualifying frames"
                }
                
                return JSONResponse(content={
                    "status": "success",
                    "message": "No qualifying frames found. Processed using sampled frames.",
                    "video_url": f"/output/{output_filename}",
                    "overall_classification": overall_classification,
                    "sampled_frames": {
                        "folder_url": f"/output/qualifying_frames_{timestamp}",
                        "count": len(sampled_frames_info),
                        "note": "Using sampled frames due to no qualifying frames meeting criteria",
                        "frames": sampled_frames_info
                    },
                    "video_info": {
                        "total_frames": frame_count,
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                        "fps": fps
                    }
                })
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "Could not process video - no valid frames found",
                        "video_url": f"/output/{output_filename}",
                        "details": {
                            "original_criteria": {
                                "min_confidence": 0.8,
                                "min_bbox_width": min_bbox_width,
                                "min_bbox_height": min_bbox_height,
                                "shape_requirements": {
                                    "vertical_orientation": True,
                                    "min_height_to_width_ratio": min_aspect_ratio
                                }
                            }
                        }
                    }
                )

        # Normal case - qualifying frames were found
        referral_count = sum(1 for frame in qualifying_frames_info 
                           if frame["resnet_classification"]["predicted_label"] == "referral")
        overall_classification = {
            "predicted_label": "referral" if referral_count >= len(qualifying_frames_info)/2 else "non-referral",
            "confidence_score": referral_count / len(qualifying_frames_info) if qualifying_frames_info else 0,
            "referral_frame_count": referral_count,
            "total_qualifying_frames": len(qualifying_frames_info)
        }

        return JSONResponse(content={
            "status": "success",
            "message": "Video processing completed with qualifying frames",
            "video_url": f"/output/{output_filename}",
            "overall_classification": overall_classification,
            "qualifying_frames": {
                "folder_url": f"/output/qualifying_frames_{timestamp}",
                "count": qualifying_frames,
                "criteria": {
                    "min_confidence": 0.8,
                    "min_bbox_width": min_bbox_width,
                    "min_bbox_height": min_bbox_height,
                    "shape_requirements": {
                        "vertical_orientation": True,
                        "min_height_to_width_ratio": min_aspect_ratio
                    }
                },
                "frames": qualifying_frames_info
            },
            "video_info": {
                "total_frames": frame_count,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "fps": fps
            }
        })

    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.remove(temp_path)
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
            
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/detections")
async def list_detections():
    """List all detection files and folders in the output directory.

    Returns:
        dict: Contains:
            - status: Success/error status
            - detections: Grouped list of all detections (videos, images, frame folders)
            - summary: Count summary of different types of detections

    Raises:
        HTTPException: If unable to access or read the output directory
    """
    try:
        all_items = []
        
        # First, get all files and directories in the output folder
        for item in OUTPUT_DIR.iterdir():
            if item.is_file() and item.suffix in ['.jpg', '.mp4']:
                # Handle regular image and video files
                all_items.append({
                    "filename": item.name,
                    "type": "video" if item.suffix == '.mp4' else "image",
                    "url": f"/output/{item.name}",
                    "timestamp": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            elif item.is_dir() and item.name.startswith('qualifying_frames_'):
                # Handle qualifying frames folders
                frames = list(item.glob('*.jpg'))  # Get all frames in the folder
                frame_infos = [
                    {
                        "filename": frame.name,
                        "url": f"/output/{item.name}/{frame.name}",
                        "timestamp": datetime.fromtimestamp(frame.stat().st_mtime).isoformat()
                    }
                    for frame in sorted(frames)
                ]
                
                # Add folder information
                all_items.append({
                    "filename": item.name,
                    "type": "qualifying_frames_folder",
                    "url": f"/output/{item.name}",
                    "timestamp": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    "frame_count": len(frames),
                    "frames": frame_infos
                })
        
        # Sort all items by timestamp
        sorted_items = sorted(all_items, key=lambda x: x["timestamp"], reverse=True)
        
        # Group items by type for better organization
        grouped_detections = {
            "videos": [item for item in sorted_items if item["type"] == "video"],
            "images": [item for item in sorted_items if item["type"] == "image"],
            "qualifying_frames_folders": [item for item in sorted_items if item["type"] == "qualifying_frames_folder"]
        }
        
        return {
            "status": "success",
            "detections": grouped_detections,
            "summary": {
                "total_videos": len(grouped_detections["videos"]),
                "total_images": len(grouped_detections["images"]),
                "total_folders": len(grouped_detections["qualifying_frames_folders"])
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    """Main entry point for running the FastAPI application."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)