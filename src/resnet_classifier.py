import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
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