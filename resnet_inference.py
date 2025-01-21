import os
import cv2
import numpy as np
from glob import glob
from keras._tf_keras.keras.models import load_model

class ResNetInference:
    def __init__(self, model_path, input_shape=(224, 224)):
        """
        Initialize the ResNet inference class
        Args:
            model_path: Path to the saved model file
            input_shape: Tuple of (height, width) for input images
        """
        self.model = load_model(model_path)
        self.height, self.width = input_shape
        self.class_names = ['non-referral', 'referral']
        
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference
        Args:
            image_path: Path to the image file
        Returns:
            Preprocessed image array and original image
        """
        # Read image in BGR (OpenCV default)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        original_img = img.copy()
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to the model's expected input shape
        img = cv2.resize(img, (self.width, self.height))
            
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img, original_img

    def predict_single_image(self, image_path, save_output=False, output_dir=None):
        """
        Perform inference on a single image
        Args:
            image_path: Path to the input image
            save_output: Whether to save the visualization
            output_dir: Directory to save the visualization
        Returns:
            Dictionary containing the prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Preprocess image
        processed_img, original_img = self.preprocess_image(image_path)
            
        # Make prediction
        prediction = self.model.predict(processed_img, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = int(prediction >= 0.5)  # Binary classification threshold
        confidence = float(prediction) if predicted_class == 1 else float(1 - prediction)
        
        # Create visualization if requested
        if save_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a visualization with prediction overlay
            vis_img = original_img.copy()
            text = f"{self.class_names[predicted_class]}: {confidence:.2f}"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)  # Green for non-referral, Red for referral
            
            # Add text to image
            cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Save visualization
            output_path = os.path.join(output_dir, f"pred_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, vis_img)
            
        return {
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': {
                'non-referral': float(1 - prediction),
                'referral': float(prediction)
            }
        }
    
    def predict_from_directory(self, input_dir, save_output=False, output_dir=None):
        """
        Perform inference on all images in a directory
        Args:
            input_dir: Directory containing input images
            save_output: Whether to save the visualizations
            output_dir: Directory to save the visualizations
        Returns:
            Dictionary containing predictions for each image
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Directory not found: {input_dir}")
            
        # Get all image files
        image_files = glob(os.path.join(input_dir, "*.jpg")) + \
                     glob(os.path.join(input_dir, "*.png"))
        
        results = {}
        for image_path in image_files:
            filename = os.path.basename(image_path)
            try:
                result = self.predict_single_image(
                    image_path, 
                    save_output=save_output,
                    output_dir=output_dir
                )
                results[filename] = result
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
                
        return results

def main():
    # Initialize the model
    model_path = "models/sankeerthana_resnet50_model.hdf5"
    classifier = ResNetInference(model_path)
    
    # Single image prediction example
    image_path = "/home/shaunliew/endoinsight-ai/data/frame_0853_png.rf.7304a4e81c8c6b6259014475221353e9.jpg"
    if os.path.exists(image_path):
        result = classifier.predict_single_image(
            image_path,
            save_output=True,
            output_dir="output"
        )
        print("\nSingle Image Prediction:")
        print(f"Predicted Class: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Class Probabilities:", result['class_probabilities'])
        
    # # Directory prediction example
    # test_dir = "/home/shaunliew/AILaryngeal/Dataset/BAGLS/test"
    # if os.path.exists(test_dir):
    #     print("\nBatch Predictions:")
    #     results = classifier.predict_from_directory(
    #         test_dir,
    #         save_output=True,
    #         output_dir="output_predictions"
    #     )
    #     for filename, result in results.items():
    #         print(f"\nFile: {filename}")
    #         print(f"Predicted Class: {result['predicted_label']}")
    #         print(f"Confidence: {result['confidence']:.4f}")
    #         print("Class Probabilities:", result['class_probabilities'])

if __name__ == "__main__":
    main()