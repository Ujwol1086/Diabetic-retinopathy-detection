"""
Advanced Image Preprocessing Pipeline for Diabetic Retinopathy Detection
This module handles comprehensive preprocessing of retinal fundus images.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Optional, List, Dict
import logging
from skimage import filters, morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
import albumentations as A

logger = logging.getLogger(__name__)

class RetinalImagePreprocessor:
    """
    Comprehensive preprocessing pipeline for retinal fundus images.
    
    Handles:
    - Image quality assessment
    - Artifact removal
    - Contrast enhancement
    - Vessel segmentation
    - Lesion detection preprocessing
    - Data augmentation
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def preprocess_image(self, image_path: str, 
                        enhance_contrast: bool = True,
                        remove_artifacts: bool = True,
                        segment_vessels: bool = False) -> Dict[str, np.ndarray]:
        """
        Comprehensive preprocessing of a retinal image.
        
        Args:
            image_path: Path to the retinal image
            enhance_contrast: Whether to apply contrast enhancement
            remove_artifacts: Whether to remove artifacts
            segment_vessels: Whether to segment blood vessels
            
        Returns:
            Dictionary containing processed images and metadata
        """
        # Load image
        image = self._load_image(image_path)
        
        # Quality assessment
        quality_score = self._assess_image_quality(image)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size)
        
        # Remove artifacts if requested
        if remove_artifacts:
            image = self._remove_artifacts(image)
        
        # Enhance contrast if requested
        if enhance_contrast:
            image = self._enhance_contrast(image)
        
        # Segment vessels if requested
        vessel_mask = None
        if segment_vessels:
            vessel_mask = self._segment_vessels(image)
        
        # Create different versions for different analysis
        processed_images = {
            'original': image,
            'enhanced': self._enhance_contrast(image),
            'clahe': self._apply_clahe(image),
            'green_channel': self._extract_green_channel(image),
            'vessel_enhanced': self._enhance_vessels(image) if vessel_mask is not None else image
        }
        
        # Add vessel mask if available
        if vessel_mask is not None:
            processed_images['vessel_mask'] = vessel_mask
        
        # Normalize all images
        for key in processed_images:
            if key != 'vessel_mask':
                processed_images[key] = self._normalize_image(processed_images[key])
        
        return {
            'images': processed_images,
            'quality_score': quality_score,
            'metadata': {
                'original_size': image.shape,
                'target_size': self.target_size,
                'preprocessing_applied': {
                    'contrast_enhancement': enhance_contrast,
                    'artifact_removal': remove_artifacts,
                    'vessel_segmentation': segment_vessels
                }
            }
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality metrics."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate quality metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate sharpness using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness = np.mean(gradient_magnitude)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate overall quality score (0-1)
        quality_score = min(1.0, (laplacian_var / 1000 + sharpness / 100 + contrast / 100) / 3)
        
        return {
            'overall_score': quality_score,
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'laplacian_variance': float(laplacian_var)
        }
    
    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Remove common artifacts from retinal images."""
        # Convert to different color spaces for artifact detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Remove bright artifacts (reflections, light spots)
        bright_mask = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30)
        image[bright_mask] = [0, 0, 0]  # Replace with black
        
        # Remove dark artifacts (shadows, vignetting)
        dark_mask = hsv[:, :, 2] < 20
        image[dark_mask] = [0, 0, 0]  # Replace with black
        
        # Apply median filtering to reduce noise
        image = cv2.medianBlur(image, 3)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using multiple techniques."""
        # Apply CLAHE to each channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        """Extract and enhance green channel (most informative for retinal images)."""
        green = image[:, :, 1]
        
        # Apply CLAHE to green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green_enhanced = clahe.apply(green)
        
        # Convert back to 3-channel
        green_3channel = np.stack([green_enhanced] * 3, axis=2)
        
        return green_3channel
    
    def _segment_vessels(self, image: np.ndarray) -> np.ndarray:
        """Segment blood vessels using Frangi filter."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Frangi filter for vessel enhancement
        from skimage.filters import frangi
        vessel_enhanced = frangi(gray, scale_range=(1, 10), scale_step=2)
        
        # Threshold to create binary mask
        vessel_mask = vessel_enhanced > np.percentile(vessel_enhanced, 90)
        
        # Morphological operations to clean up the mask
        vessel_mask = morphology.remove_small_objects(vessel_mask, min_size=50)
        vessel_mask = morphology.binary_closing(vessel_mask, morphology.disk(2))
        
        return vessel_mask.astype(np.uint8) * 255
    
    def _enhance_vessels(self, image: np.ndarray) -> np.ndarray:
        """Enhance vessel visibility in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply top-hat transform to enhance vessels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Enhance the image
        enhanced = cv2.add(gray, tophat)
        
        # Convert back to 3-channel
        enhanced_3channel = np.stack([enhanced] * 3, axis=2)
        
        return enhanced_3channel
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create data augmentation pipeline for training."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            )
        ])
    
    def augment_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image
            mask: Optional mask for segmentation tasks
            
        Returns:
            Dictionary containing augmented image and mask
        """
        if mask is not None:
            augmented = self.augmentation_pipeline(image=image, mask=mask)
            return {
                'image': augmented['image'],
                'mask': augmented['mask']
            }
        else:
            augmented = self.augmentation_pipeline(image=image)
            return {
                'image': augmented['image']
            }
    
    def preprocess_batch(self, image_paths: List[str], 
                        **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths
            **kwargs: Additional preprocessing options
            
        Returns:
            List of preprocessing results
        """
        results = []
        
        for path in image_paths:
            try:
                result = self.preprocess_image(path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {str(e)}")
                continue
        
        return results
    
    def create_lesion_maps(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create lesion detection maps for different types of diabetic retinopathy lesions.
        
        Args:
            image: Preprocessed retinal image
            
        Returns:
            Dictionary containing different lesion maps
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Microaneurysms detection (small bright spots)
        microaneurysms = self._detect_microaneurysms(gray)
        
        # Hemorrhages detection (dark spots)
        hemorrhages = self._detect_hemorrhages(gray)
        
        # Exudates detection (bright, irregular shapes)
        exudates = self._detect_exudates(gray)
        
        # Cotton wool spots detection
        cotton_wool = self._detect_cotton_wool_spots(gray)
        
        return {
            'microaneurysms': microaneurysms,
            'hemorrhages': hemorrhages,
            'exudates': exudates,
            'cotton_wool_spots': cotton_wool,
            'combined_lesions': self._combine_lesion_maps([
                microaneurysms, hemorrhages, exudates, cotton_wool
            ])
        }
    
    def _detect_microaneurysms(self, gray: np.ndarray) -> np.ndarray:
        """Detect microaneurysms using morphological operations."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Top-hat transform to enhance small bright objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _detect_hemorrhages(self, gray: np.ndarray) -> np.ndarray:
        """Detect hemorrhages using adaptive thresholding."""
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def _detect_exudates(self, gray: np.ndarray) -> np.ndarray:
        """Detect exudates using intensity-based segmentation."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold for bright regions
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def _detect_cotton_wool_spots(self, gray: np.ndarray) -> np.ndarray:
        """Detect cotton wool spots using edge detection."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        return edges
    
    def _combine_lesion_maps(self, lesion_maps: List[np.ndarray]) -> np.ndarray:
        """Combine multiple lesion maps into one."""
        combined = np.zeros_like(lesion_maps[0])
        for i, lesion_map in enumerate(lesion_maps):
            combined = cv2.add(combined, lesion_map)
        
        # Normalize
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
        
        return combined
