"""
YOLOv11 Detection Module for Player and Ball Detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict, Any
import config


class PlayerDetector:
    """YOLOv11-based detector for players and ball in sports footage"""
    
    def __init__(self, model_path: str = config.DEFAULT_MODEL_PATH):
        """
        Initialize the detector with YOLOv11 model
        
        Args:
            model_path: Path to the YOLOv11 model weights
        """
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu'
        self.model.to(self.device)
        
        # Detection parameters
        self.conf_threshold = config.DETECTION_CONFIDENCE_THRESHOLD
        self.nms_threshold = config.DETECTION_NMS_THRESHOLD
        self.target_classes = config.DETECTION_CLASSES
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect players and ball in a single frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detections with bounding boxes, confidence, and class
        """
        # Run YOLOv11 inference
        results = self.model(frame, 
                           conf=self.conf_threshold,
                           iou=self.nms_threshold,
                           classes=self.target_classes,
                           verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Convert to integer coordinates
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Determine object type
                    object_type = "ball" if class_id == 32 else "player"
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'object_type': object_type,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def filter_detections(self, detections: List[Dict[str, Any]], 
                         min_area: int = 1000) -> List[Dict[str, Any]]:
        """
        Filter detections based on area and other criteria
        
        Args:
            detections: List of detections
            min_area: Minimum bounding box area to keep
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for det in detections:
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Filter by area
            if area >= min_area:
                filtered.append(det)
        
        return filtered
    
    def get_player_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract only player detections"""
        return [det for det in detections if det['object_type'] == 'player']
    
    def get_ball_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract only ball detections"""
        return [det for det in detections if det['object_type'] == 'ball']
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn bounding boxes
        """
        frame_copy = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            object_type = det['object_type']
            confidence = det['confidence']
            
            # Get color based on object type
            color = config.BOUNDING_BOX_COLORS[object_type]
            
            # Draw bounding box
            cv2.rectangle(frame_copy, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 
                         2)
            
            # Draw label
            label = f"{object_type}: {confidence:.2f}"
            cv2.putText(frame_copy, 
                       label, 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       config.FONT_SCALE, 
                       color, 
                       config.FONT_THICKNESS)
        
        return frame_copy 