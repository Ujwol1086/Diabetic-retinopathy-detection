"""
Training Script for Diabetic Retinopathy CNN Model
This script handles model training, validation, and evaluation.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import logging
from pathlib import Path
import json
import pickle

from cnn_model import DiabeticRetinopathyCNN
from image_preprocessing import RetinalImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiabeticRetinopathyTrainer:
    """
    Trainer class for diabetic retinopathy CNN model.
    """
    
    def __init__(self, 
                 data_dir: str,
                 model_save_dir: str = "models/saved_models",
                 input_size: Tuple[int, int] = (512, 512),
                 num_classes: int = 5,
                 batch_size: int = 16,
                 epochs: int = 100):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing training data
            model_save_dir: Directory to save trained models
            input_size: Input image size
            num_classes: Number of severity classes
            batch_size: Training batch size
            epochs: Number of training epochs
        """
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.input_size = input_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Create directories
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = RetinalImagePreprocessor(target_size=input_size)
        self.model = None
        self.history = None
        
        # Class names for diabetic retinopathy severity levels
        self.class_names = [
            "No Diabetic Retinopathy",
            "Mild Non-proliferative DR",
            "Moderate Non-proliferative DR",
            "Severe Non-proliferative DR",
            "Proliferative DR"
        ]
    
    def load_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and preprocess training data.
        
        Args:
            csv_path: Path to CSV file containing image paths and labels
            
        Returns:
            Tuple of (images, labels, image_paths)
        """
        logger.info("Loading training data...")
        
        # Load metadata
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Extract image paths and labels
        image_paths = df['image_path'].tolist()
        labels = df['severity_level'].values
        
        # Preprocess images
        images = []
        valid_indices = []
        
        for i, path in enumerate(image_paths):
            try:
                # Full path
                full_path = self.data_dir / path
                
                if not full_path.exists():
                    logger.warning(f"Image not found: {full_path}")
                    continue
                
                # Preprocess image
                result = self.preprocessor.preprocess_image(
                    str(full_path),
                    enhance_contrast=True,
                    remove_artifacts=True
                )
                
                # Use enhanced version
                image = result['images']['enhanced']
                images.append(image)
                valid_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Failed to process {path}: {str(e)}")
                continue
        
        # Filter labels for valid images
        valid_labels = labels[valid_indices]
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = to_categorical(valid_labels, num_classes=self.num_classes)
        
        logger.info(f"Successfully loaded {len(images)} images")
        logger.info(f"Class distribution: {np.sum(labels, axis=0)}")
        
        return images, labels, [image_paths[i] for i in valid_indices]
    
    def create_data_generators(self, 
                              images: np.ndarray, 
                              labels: np.ndarray,
                              validation_split: float = 0.2) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """
        Create data generators for training and validation.
        
        Args:
            images: Image data
            labels: Label data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=validation_split, random_state=42, stratify=np.argmax(labels, axis=1)
        )
        
        # Create data generators
        train_generator = self._create_data_generator(X_train, y_train, is_training=True)
        val_generator = self._create_data_generator(X_val, y_val, is_training=False)
        
        return train_generator, val_generator
    
    def _create_data_generator(self, 
                              images: np.ndarray, 
                              labels: np.ndarray, 
                              is_training: bool = True) -> tf.keras.utils.Sequence:
        """Create a data generator with augmentation."""
        
        class DataGenerator(tf.keras.utils.Sequence):
            def __init__(self, images, labels, batch_size, is_training, preprocessor):
                self.images = images
                self.labels = labels
                self.batch_size = batch_size
                self.is_training = is_training
                self.preprocessor = preprocessor
                self.indices = np.arange(len(images))
                
            def __len__(self):
                return len(self.images) // self.batch_size
            
            def __getitem__(self, idx):
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_images = self.images[batch_indices]
                batch_labels = self.labels[batch_indices]
                
                if self.is_training:
                    # Apply augmentation
                    augmented_images = []
                    for image in batch_images:
                        # Convert back to uint8 for augmentation
                        image_uint8 = (image * 255).astype(np.uint8)
                        augmented = self.preprocessor.augment_image(image_uint8)
                        augmented_images.append(augmented['image'].astype(np.float32) / 255.0)
                    
                    batch_images = np.array(augmented_images)
                
                return batch_images, batch_labels
            
            def on_epoch_end(self):
                if self.is_training:
                    np.random.shuffle(self.indices)
        
        return DataGenerator(images, labels, self.batch_size, is_training, self.preprocessor)
    
    def train_model(self, 
                   images: np.ndarray, 
                   labels: np.ndarray,
                   validation_split: float = 0.2) -> Dict:
        """
        Train the CNN model.
        
        Args:
            images: Training images
            labels: Training labels
            validation_split: Fraction of data for validation
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting model training...")
        
        # Create model
        self.model = DiabeticRetinopathyCNN(
            input_shape=(*self.input_size, 3),
            num_classes=self.num_classes
        )
        model = self.model.build_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(images, labels, validation_split)
        
        # Get callbacks
        callbacks = self.model.get_callbacks(
            model_path=str(self.model_save_dir / "best_model.h5")
        )
        
        # Train model
        logger.info(f"Training for {self.epochs} epochs...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        # Save final model
        model_path = self.model_save_dir / "final_model.h5"
        model.save(str(model_path))
        logger.info(f"Final model saved to {model_path}")
        
        # Save training history
        history_path = self.model_save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def evaluate_model(self, 
                      test_images: np.ndarray, 
                      test_labels: np.ndarray) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_images: Test images
            test_labels: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info("Evaluating model...")
        
        # Make predictions
        predictions = self.model.model.predict(test_images, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Per-class accuracy
        per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        
        evaluation_results = {
            'overall_accuracy': float(accuracy),
            'per_class_accuracy': {
                self.class_names[i]: float(acc) for i, acc in enumerate(per_class_accuracy)
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'true_labels': true_classes.tolist(),
            'predicted_labels': predicted_classes.tolist()
        }
        
        # Save evaluation results
        eval_path = self.model_save_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Overall accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-2 Accuracy
        if 'top_2_accuracy' in self.history:
            axes[1, 0].plot(self.history['top_2_accuracy'], label='Training Top-2 Accuracy')
            axes[1, 0].plot(self.history['val_top_2_accuracy'], label='Validation Top-2 Accuracy')
            axes[1, 0].set_title('Top-2 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-2 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in self.history:
            axes[1, 1].plot(self.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, 
                             test_images: np.ndarray, 
                             test_labels: np.ndarray,
                             save_path: str = None):
        """Plot confusion matrix."""
        if self.model is None:
            logger.warning("Model not trained.")
            return
        
        # Get predictions
        predictions = self.model.model.predict(test_images, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Create confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_model_info(self, 
                       training_data_info: Dict,
                       model_config: Dict):
        """Save model information and configuration."""
        model_info = {
            'model_architecture': {
                'input_shape': (*self.input_size, 3),
                'num_classes': self.num_classes,
                'class_names': self.class_names
            },
            'training_config': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'input_size': self.input_size
            },
            'training_data': training_data_info,
            'model_config': model_config,
            'preprocessing': {
                'target_size': self.input_size,
                'contrast_enhancement': True,
                'artifact_removal': True
            }
        }
        
        info_path = self.model_save_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model information saved to {info_path}")


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_dir': 'data/training',
        'csv_path': 'data/training_metadata.csv',
        'model_save_dir': 'models/saved_models',
        'input_size': (512, 512),
        'num_classes': 5,
        'batch_size': 16,
        'epochs': 50
    }
    
    # Initialize trainer
    trainer = DiabeticRetinopathyTrainer(
        data_dir=config['data_dir'],
        model_save_dir=config['model_save_dir'],
        input_size=config['input_size'],
        num_classes=config['num_classes'],
        batch_size=config['batch_size'],
        epochs=config['epochs']
    )
    
    try:
        # Load data
        images, labels, image_paths = trainer.load_data(config['csv_path'])
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=np.argmax(labels, axis=1)
        )
        
        # Train model
        history = trainer.train_model(X_train, y_train, validation_split=0.2)
        
        # Evaluate model
        evaluation_results = trainer.evaluate_model(X_test, y_test)
        
        # Plot results
        trainer.plot_training_history(str(trainer.model_save_dir / "training_history.png"))
        trainer.plot_confusion_matrix(X_test, y_test, str(trainer.model_save_dir / "confusion_matrix.png"))
        
        # Save model information
        training_data_info = {
            'total_samples': len(images),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': np.sum(labels, axis=0).tolist()
        }
        
        model_config = {
            'architecture': 'EfficientNetB3 + Custom CNN',
            'optimizer': 'Adam',
            'loss_function': 'categorical_crossentropy',
            'metrics': ['accuracy', 'top_2_accuracy']
        }
        
        trainer.save_model_info(training_data_info, model_config)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
