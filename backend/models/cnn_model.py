"""
CNN Model for Diabetic Retinopathy Detection
This module contains the CNN architecture specifically designed for retinal image analysis.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DiabeticRetinopathyCNN:
    """
    CNN model for diabetic retinopathy detection from retinal fundus images.
    
    The model uses a combination of:
    - EfficientNetB3 as backbone for feature extraction
    - Custom CNN layers for retinal-specific features
    - Attention mechanisms for lesion localization
    - Multi-class classification (5 severity levels)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (512, 512, 3), 
                 num_classes: int = 5, dropout_rate: float = 0.3):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of diabetic retinopathy severity levels (0-4)
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self) -> Model:
        """
        Build the CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='retinal_image')
        
        # Data augmentation layers
        x = self._add_augmentation_layers(inputs)
        
        # Preprocessing normalization
        x = layers.Rescaling(1./255)(x)
        
        # EfficientNetB3 backbone (pre-trained on ImageNet)
        backbone = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=x,
            input_shape=self.input_shape
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in backbone.layers[:-50]:
            layer.trainable = False
            
        x = backbone.output
        
        # Custom CNN layers for retinal-specific features
        x = self._add_retinal_specific_layers(x)
        
        # Attention mechanism for lesion localization
        x = self._add_attention_layers(x)
        
        # Global pooling and classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers for classification
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        
        # Output layer for severity classification
        outputs = layers.Dense(self.num_classes, activation='softmax', name='severity_classification')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='DiabeticRetinopathyCNN')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        self.model = model
        return model
    
    def _add_augmentation_layers(self, inputs):
        """Add data augmentation layers for training robustness."""
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        x = layers.RandomBrightness(0.1)(x)
        return x
    
    def _add_retinal_specific_layers(self, x):
        """Add CNN layers specifically designed for retinal image analysis."""
        # Additional convolutional layers for retinal features
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Dilated convolutions for capturing lesions at different scales
        x = layers.Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), dilation_rate=4, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        return x
    
    def _add_attention_layers(self, x):
        """Add attention mechanism for lesion localization."""
        # Channel attention
        channel_attention = self._channel_attention(x)
        x = layers.Multiply()([x, channel_attention])
        
        # Spatial attention
        spatial_attention = self._spatial_attention(x)
        x = layers.Multiply()([x, spatial_attention])
        
        return x
    
    def _channel_attention(self, x):
        """Channel attention mechanism."""
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        # Shared MLP
        avg_pool = layers.Dense(x.shape[-1] // 8, activation='relu')(avg_pool)
        avg_pool = layers.Dense(x.shape[-1], activation='sigmoid')(avg_pool)
        
        max_pool = layers.Dense(x.shape[-1] // 8, activation='relu')(max_pool)
        max_pool = layers.Dense(x.shape[-1], activation='sigmoid')(max_pool)
        
        attention = layers.Add()([avg_pool, max_pool])
        attention = layers.Reshape((1, 1, x.shape[-1]))(attention)
        
        return attention
    
    def _spatial_attention(self, x):
        """Spatial attention mechanism."""
        avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(x)
        max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(x)
        
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        attention = layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')(concat)
        
        return attention
    
    def get_callbacks(self, model_path: str = 'best_model.h5') -> list:
        """
        Get training callbacks.
        
        Args:
            model_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def predict_severity(self, image: np.ndarray) -> dict:
        """
        Predict diabetic retinopathy severity from retinal image.
        
        Args:
            image: Preprocessed retinal image array
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Ensure image has correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image, verbose=0)
        severity_probabilities = prediction[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(severity_probabilities)
        confidence = float(np.max(severity_probabilities))
        
        # Severity level descriptions
        severity_levels = {
            0: "No Diabetic Retinopathy",
            1: "Mild Non-proliferative Diabetic Retinopathy",
            2: "Moderate Non-proliferative Diabetic Retinopathy", 
            3: "Severe Non-proliferative Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy"
        }
        
        # Risk assessment
        risk_level = self._assess_risk_level(predicted_class, confidence)
        
        return {
            'predicted_class': int(predicted_class),
            'severity_level': severity_levels[predicted_class],
            'confidence': confidence,
            'risk_level': risk_level,
            'all_probabilities': {
                f'Level_{i}': float(prob) for i, prob in enumerate(severity_probabilities)
            }
        }
    
    def _assess_risk_level(self, predicted_class: int, confidence: float) -> str:
        """Assess risk level based on predicted class and confidence."""
        if predicted_class == 0:
            return "Low" if confidence > 0.8 else "Low-Medium"
        elif predicted_class in [1, 2]:
            return "Medium" if confidence > 0.7 else "Medium-High"
        else:
            return "High" if confidence > 0.6 else "Very High"
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet."
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class RetinalImagePreprocessor:
    """
    Image preprocessing pipeline for retinal fundus images.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a retinal image for CNN input.
        
        Args:
            image_path: Path to the retinal image
            
        Returns:
            Preprocessed image array
        """
        import cv2
        from PIL import Image
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # for better contrast in retinal images
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def preprocess_batch(self, image_paths: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batch of preprocessed images
        """
        processed_images = []
        
        for path in image_paths:
            try:
                processed_image = self.preprocess_image(path)
                processed_images.append(processed_image)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {str(e)}")
                continue
        
        return np.array(processed_images)
