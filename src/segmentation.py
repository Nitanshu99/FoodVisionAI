"""
Segmentation Engine Module.

This module integrates YOLOv8-seg to perform:
1. Plate Detection (to establish scale).
2. Food Item Segmentation (to get precise masks).
3. Scale Recovery (Pixels-Per-Metric calculation).

It replaces the 'mock' logic with real Computer Vision geometry.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import config

class DietaryAssessor:
    """
    The Geometric Intelligence Unit.
    Responsible for converting raw pixels into metric measurements (cm, cm^2).
    """

    def __init__(self):
        """
        Initializes the YOLOv8 segmentation model.
        Auto-downloads 'yolov8m-seg.pt' if not found in 'models/'.
        """
        print(f"Loading YOLOv8 Model from: {config.YOLO_MODEL_PATH}")
        # task='segment' ensures we get masks, not just boxes
        self.model = YOLO(config.YOLO_MODEL_PATH, task='segment')

    def detect_plate_and_scale(self, image):
        """
        Detects the dining plate to calculate Pixels-Per-Metric (PPM).
        
        Heuristic: 
        - The largest detected object of class 'bowl', 'plate', or similar is the container.
        - We fit an ellipse to it.
        - The Major Axis of the ellipse = Real World Diameter (28cm).
        
        Args:
            image (np.ndarray): The input image (BGR).
            
        Returns:
            float: The calculated Pixels-Per-Metric (PPM) ratio.
        """
        # Run inference (only looking for plates/bowls if possible, but standard model finds all)
        results = self.model(image, verbose=False)[0]
        
        max_area = 0
        plate_mask = None
        
        # COCO Classes for containers: 41=cup, 43=knife, 44=spoon, 45=bowl, ...
        # Standard YOLOv8 is trained on COCO. 'Bowl' is class 45. 
        # Note: A round plate often gets detected as a 'bowl' or 'frisbee' (29) or generic object.
        # For robustness, we look for the largest central object if no specific 'plate' class exists.
        
        if results.masks:
            for seg in results.masks.data:
                # Convert to numpy binary mask
                mask = seg.cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                area = np.sum(mask)
                
                # Heuristic: The plate is likely the largest object
                if area > max_area:
                    max_area = area
                    plate_mask = mask

        if plate_mask is not None:
            # 1. Extract Contour
            contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 2. Fit Ellipse
                if len(largest_contour) >= 5:
                    (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(largest_contour)
                    
                    # 3. Calculate PPM (Major Axis = Real Diameter)
                    # config.PLATE_DIAMETER_CM is 28.0
                    ppm = major_axis / config.PLATE_DIAMETER_CM
                    return ppm

        # Fallback: If no plate found, assume a standard view width
        # (e.g., 50cm field of view for a 1080p image)
        # This is a safety fallback to prevent crash.
        default_fov_width_cm = 50.0
        return image.shape[1] / default_fov_width_cm

    def analyze_scene(self, image):
        """
        Full pipeline: Detect Plate -> Get Scale -> Segment Food.
        
        Returns:
            list: A list of dictionaries, one for each detected food item.
                  [{'bbox': [x1,y1,x2,y2], 'mask': np.array, 'area_cm2': float}]
        """
        ppm = self.detect_plate_and_scale(image)
        results = self.model(image, verbose=False)[0]
        
        detected_objects = []
        
        if results.masks is None:
            return detected_objects, ppm

        # Iterate through detections
        for i, box in enumerate(results.boxes):
            # Get class ID (COCO classes)
            # We filter for likely food items or process everything that isn't background
            # For now, we process all detections and let EfficientNet filter non-foods later.
            
            # Extract Mask
            mask_raw = results.masks.data[i].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask_raw, (image.shape[1], image.shape[0]))
            
            # Extract Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Geometric Properties
            area_pixels = np.sum(mask)
            area_cm2 = area_pixels / (ppm ** 2)
            
            detected_objects.append({
                "bbox": (x1, y1, x2, y2),
                "mask": mask,
                "area_pixels": area_pixels,
                "area_cm2": area_cm2,
                "confidence": float(box.conf)
            })
            
        return detected_objects, ppm