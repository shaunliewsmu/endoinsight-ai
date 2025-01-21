from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from GhostNet.ghostNet import GhostNet
import numpy as np
import cv2
import os

class GhostNetClassifier:
    def __init__(self, model_path, num_classes=1):
        """
        Initialize GhostNet classifier
        Args:
            model_path: Path to the saved model weights
            num_classes: Number of classification classes (1 for binary with sigmoid)
        """
        self.num_classes = num_classes
        self.input_shape = (256, 256, 1)
        
        # Initialize and load model
        self.model = GhostNet(self.input_shape, self.num_classes).build()
        self.model.load_weights(model_path)
        
        # Initialize ImageDataGenerator for preprocessing
        self.data_generator = ImageDataGenerator(
            rescale=1./255,
            dtype='float32'
        )

    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for inference
        Args:
            image_path: Path to the image file
        Returns:
            Preprocessed image array
        """
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        # Resize to match input shape
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        
        # Convert to float32 and normalize before adding dimensions
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension and batch dimension
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        return img

    def predict_single_image(self, image_path):
        """
        Make prediction on a single image
        Args:
            image_path: Path to input image
        Returns:
            Dictionary containing prediction results
        """
        try:
            processed_img = self.preprocess_single_image(image_path)
            prediction = self.model.predict(processed_img, verbose=0)
            
            if self.num_classes == 1:
                probability = float(prediction[0][0])
                predicted_class = 1 if probability >= 0.5 else 0
                result = {
                    "predicted_class": predicted_class,
                    "predicted_label": "Referral" if predicted_class == 1 else "Non-Referral",
                    "confidence": probability if predicted_class == 1 else 1 - probability,
                    "class_probabilities": {
                        "class_0": float(1 - probability),
                        "class_1": float(probability)
                    }
                }
            else:
                predicted_class = np.argmax(prediction[0])
                class_probabilities = prediction[0]
                result = {
                    "predicted_class": int(predicted_class),
                    "predicted_label": "Referral" if predicted_class == 1 else "Non-Referral",
                    "confidence": float(class_probabilities[predicted_class]),
                    "class_probabilities": {
                        f"class_{i}": float(prob) 
                        for i, prob in enumerate(class_probabilities)
                    }
                }
            
            return result
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def predict_from_directory(self, directory_path, batch_size=32):
        """
        Make predictions on all images in a directory
        Args:
            directory_path: Path to directory containing images
            batch_size: Batch size for predictions
        Returns:
            Dictionary of predictions for each image
        """
        # Get all image files from directory
        image_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        results = {}
        for image_path in image_files:
            relative_path = os.path.relpath(image_path, directory_path)
            result = self.predict_single_image(image_path)
            if result is not None:
                results[relative_path] = result
            
        return results

def main():
    # Example usage
    model_path = "models/ghostNet_0923.hdf5"
    
    try:
        # Initialize classifier with num_classes=1 for binary classification
        classifier = GhostNetClassifier(model_path, num_classes=1)
        
        # Single image prediction example
        image_path = "/home/shaunliew/endoinsight-ai/data/frame_0853_png.rf.7304a4e81c8c6b6259014475221353e9.jpg"
        if os.path.exists(image_path):
            result = classifier.predict_single_image(image_path)
            if result:
                print("\nSingle Image Prediction:")
                print(f"Predicted Class: {result['predicted_label']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("Class Probabilities:", result['class_probabilities'])
        
        # Directory prediction example
        # test_dir = "/home/shaunliew/AILaryngeal/Dataset/BAGLS/test"
        # if os.path.exists(test_dir):
        #     print("\nBatch Predictions:")
        #     results = classifier.predict_from_directory(test_dir)
        #     for filename, result in results.items():
        #         print(f"\nFile: {filename}")
        #         print(f"Predicted Class: {result['predicted_label']}")
        #         print(f"Confidence: {result['confidence']:.4f}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()