from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel 
from pathlib import Path
import base64
import json
from typing import List, Dict
from datetime import datetime
import asyncio
import re

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    aspect_ratio: float

class YOLODetection(BaseModel):
    confidence: float
    bbox: BoundingBox

class ResNetClassification(BaseModel):
    predicted_label: str
    confidence: float
    class_probabilities: Dict[str, float]

class Frame(BaseModel):
    frame_number: int
    url: str
    yolo_detection: YOLODetection
    resnet_classification: ResNetClassification

class QualifyingFrames(BaseModel):
    folder_url: str
    count: int
    frames: List[Frame]

class VideoProcessingResult(BaseModel):
    status: str
    qualifying_frames: QualifyingFrames

class ConsolidatedAnalysis(BaseModel):
    """Consolidated analysis output for all qualifying frames"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    glottis_assessment: dict = Field(description="Overall glottis assessment")
    suggested_actions: List[str] = Field(description="Clinical recommendations")
    ai_performance: dict = Field(description="AI model performance metrics")

class ImageAnalyzer:
    def __init__(self, base_path: str = "/home/shaunliew/endoinsight-ai"):
        self.base_path = base_path
        self.vision_model = OllamaModel(model_name='llama3.2-vision')
        self.agent = Agent(
            model=self.vision_model,
            result_retries=3,
            system_prompt="""You are an expert head and neck cancer specialist analyzing laryngeal endoscopy images and AI detection/classification results.

CRITICAL FORMAT RULES - READ CAREFULLY:
1. Output EXACTLY ONE JSON object
2. All values MUST be simple strings or arrays of strings
3. NO nested objects or complex structures
4. NO additional text before or after JSON
5. Use ONLY the exact structure shown below

ANALYSIS CONTEXT:
1. Detection Metrics (YOLO):
   - Confidence >0.8: Reliable glottis detection
   - Aspect ratio >1.5: Proper anatomical view
   - Bounding box size: Indicates structure visibility

2. Classification Results (ResNet):
   - "referral" label: Indicates need for specialist review
   - High confidence: Suggests reliable assessment
   - Multiple referral frames: Increases urgency

REQUIRED OUTPUT FORMAT:
{
    "glottis_assessment": {
        "overall_condition": "single string describing condition and detection reliability",
        "symmetry": "single string analyzing structural symmetry with confidence context",
        "tissue_state": "single string describing tissue characteristics and classification",
        "abnormalities": [
            "string describing specific finding 1",
            "string describing specific finding 2"
        ]
    },
    "suggested_actions": [
        "string with clear referral recommendation",
        "string with specific follow-up instructions"
    ]
}

RESPONSE RULES:
1. overall_condition:
   - Include detection confidence
   - Mention structure visibility
   - Reference classification results

2. symmetry:
   - Describe structural balance
   - Reference detection metrics
   - Note any asymmetries

3. tissue_state:
   - Describe visible characteristics
   - Note any abnormal features
   - Reference classification confidence

4. abnormalities:
   - List specific findings as simple strings
   - Include detection/classification context
   - Maximum 3-4 findings

5. suggested_actions:
   - First action: Clear referral decision
   - Second action: Specific follow-up plan
   - Keep as simple strings

CRITICAL: NO NESTED OBJECTS OR COMPLEX STRUCTURES ALLOWED

Error Response Format:
{
    "glottis_assessment": {
        "overall_condition": "Analysis error - insufficient quality for assessment",
        "symmetry": "Unable to assess glottic symmetry",
        "tissue_state": "Unable to assess tissue characteristics",
        "abnormalities": ["Image quality prevents reliable assessment"]
    },
    "suggested_actions": ["Repeat imaging with improved quality"]
}"""
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def validate_json_structure(self, data: dict) -> bool:
        """Validate that the JSON response has the correct structure."""
        try:
            # Check for required top-level keys
            required_keys = {"glottis_assessment", "suggested_actions"}
            if not all(key in data for key in required_keys):
                return False
            
            # Check glottis_assessment structure
            assessment = data["glottis_assessment"]
            required_assessment_keys = {
                "overall_condition", "symmetry", "tissue_state", "abnormalities"
            }
            if not all(key in assessment for key in required_assessment_keys):
                return False
            
            # Validate types
            if not isinstance(assessment["abnormalities"], list):
                return False
            if not isinstance(data["suggested_actions"], list):
                return False
            
            # Validate string fields
            string_fields = [
                assessment["overall_condition"],
                assessment["symmetry"],
                assessment["tissue_state"]
            ]
            if not all(isinstance(field, str) for field in string_fields):
                return False
                
            return True
        except (KeyError, TypeError):
            return False

    def clean_response(self, response: str) -> str:
        """Clean and validate the response to ensure single JSON object."""
        # Remove any markdown and text
        response = response.replace('```json', '').replace('```', '')
        
        # Find all JSON objects
        json_objects = []
        bracket_count = 0
        start_idx = -1
        
        for i, char in enumerate(response):
            if char == '{':
                if bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0 and start_idx != -1:
                    json_objects.append(response[start_idx:i+1])
        
        # If we found exactly one JSON object, return it
        if len(json_objects) == 1:
            return json_objects[0]
        
        # If we found multiple objects, try to find the most complete one
        for obj in json_objects:
            try:
                parsed = json.loads(obj)
                if self.validate_json_structure(parsed):
                    return obj
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return the original response
        # (it will be handled by extract_json_from_response)
        return response

    def extract_json_from_response(self, response: str) -> dict:
        """Extract and parse JSON from the response string."""
        try:
            # First try to parse the cleaned response
            cleaned_response = self.clean_response(response)
            parsed_response = json.loads(cleaned_response)
            
            # Validate the structure
            if self.validate_json_structure(parsed_response):
                return parsed_response
                
            print("\n⚠️ Invalid JSON structure, returning error response")
            raise ValueError("Invalid JSON structure")
            
        except (json.JSONDecodeError, ValueError):
            print("\n⚠️ JSON parsing failed, returning error response")
            # Return a standardized error response
            return {
                "glottis_assessment": {
                    "overall_condition": "Error: JSON parsing failed",
                    "symmetry": "Not available",
                    "tissue_state": "Not available",
                    "abnormalities": ["Response format error"]
                },
                "suggested_actions": ["Review system response format"]
            }

    async def analyze_frames(self, frames: List[Frame], max_retries: int = 3) -> ConsolidatedAnalysis:
        """Analyze frames with retry logic and strict validation."""
        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "type": "text", 
                        "text": "Analyze these endoscopy images. Return EXACTLY ONE JSON object with NO additional text."
                    }
                ]
                
                # Add frame images
                for frame in frames:
                    image_path = str(Path(self.base_path) / frame.url.lstrip('/'))
                    if not Path(image_path).exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    
                    base64_image = self.encode_image(image_path)
                    messages.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    })

                result = await self.agent.run(messages)
                
                print(f"\n🔍 Raw AI Response (Attempt {attempt + 1}):")
                print("-" * 50)
                print(str(result.data))
                print("-" * 50)
                
                # Clean and validate response
                cleaned_response = self.clean_response(str(result.data))
                try:
                    analysis_dict = json.loads(cleaned_response)
                    if self.validate_json_structure(analysis_dict):
                        # Calculate performance metrics
                        ai_performance = {
                            "average_detection_confidence": sum(f.yolo_detection.confidence for f in frames) / len(frames),
                            "average_classification_confidence": sum(f.resnet_classification.confidence for f in frames) / len(frames),
                            "referral_frames_ratio": sum(1 for f in frames if f.resnet_classification.predicted_label == "referral") / len(frames),
                            "total_frames_analyzed": len(frames)
                        }
                        
                        return ConsolidatedAnalysis(
                            glottis_assessment=analysis_dict["glottis_assessment"],
                            suggested_actions=analysis_dict["suggested_actions"],
                            ai_performance=ai_performance
                        )
                    else:
                        print(f"\n⚠️ Attempt {attempt + 1}: Invalid JSON structure, retrying...")
                except json.JSONDecodeError:
                    print(f"\n⚠️ Attempt {attempt + 1}: Invalid JSON format, retrying...")
                    
            except Exception as e:
                print(f"\n❌ Error in attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise

        # Return error response if all retries fail
        return ConsolidatedAnalysis(
            glottis_assessment={
                "overall_condition": "Error: Maximum retries exceeded",
                "symmetry": "Not available",
                "tissue_state": "Not available",
                "abnormalities": ["Failed to get valid response after multiple attempts"]
            },
            suggested_actions=["System error - please try again later"],
            ai_performance={}
        )

async def main():
    try:
        # Load the JSON data
        with open('sample_data.json', 'r') as f:
            data = json.load(f)
        
        # Parse the video processing result
        result = VideoProcessingResult(**data)
        
        print("\n🔍 Starting consolidated analysis of qualifying frames...")
        print(f"Processing {result.qualifying_frames.count} frames")
        
        # Create analyzer and get consolidated analysis
        analyzer = ImageAnalyzer()
        analysis = await analyzer.analyze_frames(result.qualifying_frames.frames)
        
        # Print results
        print("\n📋 Consolidated Analysis Results:")
        print("="*50)
        print(json.dumps(analysis.model_dump(), indent=2))
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\n🚀 Starting medical image analysis...")
    print("Make sure Ollama is running with: ollama run llama3.2-vision")
    
    asyncio.run(main())