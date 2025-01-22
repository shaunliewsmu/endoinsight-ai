# src/llm_analyzer.py

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional
import base64
import json
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
import traceback

class GlottisAssessment(BaseModel):
    overall_condition: str
    symmetry: str
    tissue_state: str
    abnormalities: List[str]

class ConsolidatedAnalysis(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    glottis_assessment: GlottisAssessment
    suggested_actions: List[str]
    ai_performance: Dict[str, float]

class LLMAnalyzer:
    def __init__(self, base_path: str = "output"):
        """Initialize LLM Analyzer with base path for images."""
        self.base_path = Path(base_path)
        self.setup_llm()
    
    def setup_llm(self):
        """Initialize the LLM model with Ollama."""
        self.vision_model = OllamaModel(model_name='llama3.2-vision')
        self.agent = Agent(
            model=self.vision_model,
            result_retries=3,
            system_prompt="""You are an expert head and neck cancer specialist analyzing laryngeal endoscopy images and AI detection/classification results.

CRITICAL FORMAT RULES:
1. Output EXACTLY ONE JSON object
2. All values MUST be simple strings or arrays of strings
3. NO nested objects or complex structures
4. NO additional text before or after JSON
5. ONLY respond with JSON - no explanations or other text

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
}"""
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            # Clean the path: remove 'output/', '/output/' and any leading slash
            clean_path = (image_path
                .replace('output/', '')
                .replace('/output/', '')
                .lstrip('/')  # Remove leading slash
            )
            
            # Construct the full path using the base path
            full_path = self.base_path / clean_path
            
            # Debug print
            print(f"Reading image from: {full_path}")
            
            # Read and encode the image
            with open(full_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
                
        except FileNotFoundError as e:
            print(f"File not found error:")
            print(f"Attempted path: {full_path}")
            print(f"Base path: {self.base_path}")
            print(f"Input path: {image_path}")
            print(f"Cleaned path: {clean_path}")
            raise

    def clean_response(self, response: str) -> str:
        """Clean and validate LLM response."""
        try:
            # Print raw response for debugging
            print("\nRaw LLM Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Remove markdown formatting and whitespace
            response = response.replace('```json', '').replace('```', '').strip()
            
            # Find JSON object
            start = response.find('{')
            end = response.rfind('}')
            
            if start == -1 or end == -1:
                print("No JSON brackets found in response")
                raise ValueError("No valid JSON found")
                
            json_str = response[start:end+1]
            
            # Attempt to parse and validate JSON structure
            parsed = json.loads(json_str)
            
            # Validate required structure
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")
                
            required_keys = ["glottis_assessment", "suggested_actions"]
            if not all(key in parsed for key in required_keys):
                raise ValueError(f"Missing required keys: {set(required_keys) - set(parsed.keys())}")
                
            assessment_keys = ["overall_condition", "symmetry", "tissue_state", "abnormalities"]
            if not all(key in parsed["glottis_assessment"] for key in assessment_keys):
                raise ValueError(f"Missing assessment keys: {set(assessment_keys) - set(parsed['glottis_assessment'].keys())}")
                
            return json_str
            
        except Exception as e:
            print(f"Error cleaning response: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return self._get_error_json()

    def _get_error_json(self) -> str:
        """Return error JSON when response processing fails."""
        return json.dumps({
            "glottis_assessment": {
                "overall_condition": "Error in analysis",
                "symmetry": "Not available",
                "tissue_state": "Not available",
                "abnormalities": ["Analysis failed"]
            },
            "suggested_actions": ["Please try again"]
        })

    async def analyze_frames(self, frames_data: List[dict]) -> ConsolidatedAnalysis:
        """Analyze frames and return consolidated analysis."""
        try:
            print(f"\nAnalyzing {len(frames_data)} frames...")
            
            # Calculate AI performance metrics
            ai_performance = {
                "average_detection_confidence": sum(f["yolo_detection"]["confidence"] 
                    for f in frames_data) / len(frames_data),
                "average_classification_confidence": sum(f["resnet_classification"]["confidence"] 
                    for f in frames_data) / len(frames_data),
                "referral_frames_ratio": sum(1 for f in frames_data 
                    if f["resnet_classification"]["predicted_label"] == "referral") / len(frames_data),
                "total_frames_analyzed": len(frames_data)
            }

            # Prepare messages for LLM
            messages = [
                {
                    "type": "text",
                    "text": "Analyze these endoscopy images and respond ONLY with the required JSON format."
                }
            ]

            # Add frame images
            valid_images = 0
            for frame in frames_data:
                try:
                    encoded_image = self.encode_image(frame['url'])
                    messages.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "high"
                        }
                    })
                    valid_images += 1
                except Exception as e:
                    print(f"Error processing frame {frame['url']}: {str(e)}")
                    continue

            if valid_images == 0:
                raise ValueError("No valid images to analyze")

            print(f"Successfully encoded {valid_images} images for analysis")

            # Get LLM analysis
            result = await self.agent.run(messages)
            llm_result = json.loads(self.clean_response(str(result.data)))

            # Create and return consolidated analysis
            return ConsolidatedAnalysis(
                glottis_assessment=GlottisAssessment(**llm_result["glottis_assessment"]),
                suggested_actions=llm_result["suggested_actions"],
                ai_performance=ai_performance
            )

        except Exception as e:
            print(f"Error in LLM analysis: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return self._get_error_analysis()

    def _get_error_analysis(self) -> ConsolidatedAnalysis:
        """Return error analysis when processing fails."""
        return ConsolidatedAnalysis(
            glottis_assessment=GlottisAssessment(
                overall_condition="Error in analysis",
                symmetry="Not available",
                tissue_state="Not available",
                abnormalities=["Analysis failed"]
            ),
            suggested_actions=["Please try again"],
            ai_performance={}
        )