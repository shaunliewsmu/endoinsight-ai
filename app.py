# Import necessary modules
from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

# Import the custom modules
from src.config import WEIGHTS_PATH, DATA_PATH, DEVICE
from src.yolo_detector import YOLODetector
from src.resnet_classifier import ResNetClassifier
from src.llm_analyzer import LLMAnalyzer

def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def prepare_for_json(data):
    """Prepare MongoDB document for JSON serialization."""
    if isinstance(data, dict):
        return {k: prepare_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [prepare_for_json(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, ObjectId):
        return str(data)
    return data



app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an APIRouter
api_router = APIRouter()

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount the output directory to make images accessible via URL
app.mount("/output", StaticFiles(directory="output"), name="output")

# Initialize MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['hnc_detection']
collection = db['video_analysis']

# Initialize the YOLO detector
detector = YOLODetector(
    weights_path=WEIGHTS_PATH,
    data_path=DATA_PATH,
    device=DEVICE
)

# Initialize ResNet classifier
resnet_classifier = ResNetClassifier("models/sankeerthana_resnet50_model.hdf5")

# Initialize LLM Analyzer
llm_analyzer = LLMAnalyzer()

@app.get("/")
async def root():
    """Root endpoint to check API status.

    Returns:
        dict: Status message and operational status
    """
    return {"message": "Welcome to the Head and Neck Cancer Glottis Detection API", "status": "operational"}

@api_router.post("/image-detect-glottis")
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

@api_router.post("/video-analyze")
async def video_analyze(file: UploadFile = File(...)):
    """Process video with YOLO detection, ResNet classification, and LLM analysis."""
    try:
        # First, process the video using existing logic
        video_result = await process_video(file)
        
        # Convert video_result to dict if it's a JSONResponse
        if isinstance(video_result, JSONResponse):
            video_data = video_result.body
            if isinstance(video_data, bytes):
                video_data = json.loads(video_data)
        else:
            video_data = video_result
            
        if video_data.get("status") != "success":
            return JSONResponse(content=video_data)
        
        # Check if we have qualifying frames
        if "qualifying_frames" not in video_data:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No qualifying frames found for analysis",
                    **video_data
                }
            )
        
        # Perform LLM analysis on qualifying frames
        llm_analysis = await llm_analyzer.analyze_frames(
            video_data["qualifying_frames"]["frames"]
        )
        
        # Combine results
        final_result = {
            **video_data,
            "llm_analysis": llm_analysis.model_dump(),
            "created_at": datetime.now()
        }
        
        # Save to MongoDB
        inserted_id = collection.insert_one(final_result).inserted_id
        
        # Prepare response by converting MongoDB document for JSON serialization
        final_result = prepare_for_json(final_result)
        final_result['_id'] = str(inserted_id)
        
        return JSONResponse(content=final_result)
        
    except Exception as e:
        import traceback
        print("Error in video_analyze:")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }
        )

async def process_video(file: UploadFile):
    """Process video file and extract qualifying frames."""
    temp_path = None
    cap = None
    out = None
    
    try:
        # Save uploaded video temporarily
        temp_path = OUTPUT_DIR / f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # Open video file
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            return {
                "status": "error",
                "message": "Could not open video file"
            }

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

        # Try different codecs in order of preference
        codecs = ['avc1', 'mp4v', 'H264', 'XVID']
        out = None
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (frame_width, frame_height),
                    True  # isColor
                )
                if out.isOpened():
                    print(f"Successfully initialized VideoWriter with codec: {codec}")
                    break
            except Exception as e:
                print(f"Failed to initialize VideoWriter with codec {codec}: {e}")
                if out is not None:
                    out.release()
                continue
        
        if out is None or not out.isOpened():
            raise Exception("Failed to initialize video writer with any codec")

        frame_count = 0
        qualifying_frames = 0
        qualifying_frames_info = []

        print(f"Starting video processing: {total_frames} frames total")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with YOLO
            result = detector.process_image(frame)
            success = out.write(result["annotated_image"])
            
            if not success:
                print(f"Warning: Failed to write frame {frame_count}")
            
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
                        
                        # Store frame info
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
                            "resnet_classification": resnet_result
                        })
                        qualifying_frames += 1
                    break

            frame_count += 1
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")

        # Release resources
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

        # Verify the output video
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"Video saved successfully. Size: {file_size} bytes")
            
            if file_size == 0:
                raise Exception("Output video file is empty")
        else:
            raise Exception("Output video file was not created")

        # Try to convert video to web-compatible format using FFmpeg if available
        try:
            import ffmpeg
            print("Converting video to web-compatible format...")
            
            stream = ffmpeg.input(str(output_path))
            stream = ffmpeg.output(
                stream, 
                str(output_path.with_suffix('.web.mp4')),
                vcodec='libx264',
                acodec='aac',
                **{'b:v': '2M'}  # 2 Mbps bitrate
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
            
            # Replace original with converted file
            os.replace(output_path.with_suffix('.web.mp4'), output_path)
            print("Video conversion completed successfully")
            
        except ImportError:
            print("FFmpeg Python bindings not available, skipping conversion")
        except Exception as e:
            print(f"FFmpeg conversion failed: {str(e)}")
            # Continue with original file if conversion fails

        # Clean up temporary file
        if temp_path is not None and temp_path.exists():
            os.remove(temp_path)

        # Return results
        if qualifying_frames_info:
            return {
                "status": "success",
                "message": "Video processing completed with qualifying frames",
                "video_url": f"/output/{output_filename}",
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
            }
        else:
            return {
                "status": "error",
                "message": "No qualifying frames found",
                "video_url": f"/output/{output_filename}",
                "video_info": {
                    "total_frames": frame_count,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "fps": fps
                }
            }

    except Exception as e:
        import traceback
        print("Error in process_video:")
        print(traceback.format_exc())
        
        # Clean up resources
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if temp_path is not None and temp_path.exists():
            os.remove(temp_path)
            
        return {
            "status": "error",
            "message": str(e)
        }

                
@api_router.get("/analysis")
async def list_all_analysis():
    """List all detection results from MongoDB and filesystem."""
    try:
        # Get all documents from MongoDB
        mongo_results = []
        for doc in collection.find().sort("created_at", -1):
            # Prepare document for JSON serialization
            prepared_doc = prepare_for_json(doc)
            mongo_results.append(prepared_doc)
        
        # Get filesystem items (for compatibility with existing files)
        all_items = []
        for item in OUTPUT_DIR.iterdir():
            if item.is_file() and item.suffix in ['.jpg', '.mp4']:
                all_items.append({
                    "filename": item.name,
                    "type": "video" if item.suffix == '.mp4' else "image",
                    "url": f"/output/{item.name}",
                    "timestamp": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            elif item.is_dir() and item.name.startswith('qualifying_frames_'):
                frames = list(item.glob('*.jpg'))
                frame_infos = [
                    {
                        "filename": frame.name,
                        "url": f"/output/{item.name}/{frame.name}",
                        "timestamp": datetime.fromtimestamp(frame.stat().st_mtime).isoformat()
                    }
                    for frame in sorted(frames)
                ]
                
                all_items.append({
                    "filename": item.name,
                    "type": "qualifying_frames_folder",
                    "url": f"/output/{item.name}",
                    "timestamp": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    "frame_count": len(frames),
                    "frames": frame_infos
                })
        
        # Sort filesystem items by timestamp
        sorted_items = sorted(all_items, key=lambda x: x["timestamp"], reverse=True)
        
        # Group items
        grouped_detections = {
            "analysis_results": mongo_results,  # Add MongoDB results
            "videos": [item for item in sorted_items if item["type"] == "video"],
            "images": [item for item in sorted_items if item["type"] == "image"],
            "qualifying_frames_folders": [item for item in sorted_items if item["type"] == "qualifying_frames_folder"]
        }
        
        return {
            "status": "success",
            "detections": grouped_detections,
            "summary": {
                "total_analyses": len(mongo_results),
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

@api_router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve a specific analysis result by ID.
    
    Args:
        analysis_id (str): MongoDB ObjectId as string
    
    Returns:
        JSONResponse: The complete analysis result or error message
    """
    try:
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(analysis_id)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Invalid analysis ID format"
                }
            )
        
        # Query MongoDB for the specific document
        result = collection.find_one({"_id": obj_id})
        
        # Check if document exists
        if result is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Analysis with ID {analysis_id} not found"
                }
            )
        
        # Prepare document for JSON serialization
        prepared_result = prepare_for_json(result)
        
        return JSONResponse(content={
            "status": "success",
            "analysis": prepared_result
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to retrieve analysis: {str(e)}"
            }
        )

# Include the router with a prefix
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    """Main entry point for running the FastAPI application."""
    import uvicorn
    print("Starting server with Ollama and MongoDB integration...")
    uvicorn.run(app, host="0.0.0.0", port=8000)