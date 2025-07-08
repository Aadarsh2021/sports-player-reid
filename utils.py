"""
Utility functions for Player Re-Identification System
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import config


def extract_appearance_features(frame: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Extract appearance features from a player bounding box
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Feature vector representing player appearance
    """
    # Extract player region
    x1, y1, x2, y2 = bbox
    player_region = frame[y1:y2, x1:x2]
    
    if player_region.size == 0:
        return np.zeros(config.REID_FEATURE_DIM)
    
    # Resize to standard size
    player_region = cv2.resize(player_region, (64, 128))
    
    # Convert to HSV color space for better feature representation
    hsv_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
    
    # Extract color histogram features
    hist_features = []
    
    # H channel histogram (8 bins)
    h_hist = cv2.calcHist([hsv_region], [0], None, [8], [0, 180])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    hist_features.extend(h_hist)
    
    # S channel histogram (8 bins)
    s_hist = cv2.calcHist([hsv_region], [1], None, [8], [0, 256])
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    hist_features.extend(s_hist)
    
    # V channel histogram (8 bins)
    v_hist = cv2.calcHist([hsv_region], [2], None, [8], [0, 256])
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    hist_features.extend(v_hist)
    
    # Extract spatial features (simple grid-based features)
    spatial_features = []
    grid_size = 4
    h, w = player_region.shape[:2]
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract grid cell
            cell_h_start = i * h // grid_size
            cell_h_end = (i + 1) * h // grid_size
            cell_w_start = j * w // grid_size
            cell_w_end = (j + 1) * w // grid_size
            
            cell = player_region[cell_h_start:cell_h_end, cell_w_start:cell_w_end]
            
            # Calculate mean color for this cell
            mean_color = np.mean(cell, axis=(0, 1))
            spatial_features.extend(mean_color)
    
    # Combine all features
    features = np.concatenate([hist_features, spatial_features])
    
    # Normalize to unit length
    if np.linalg.norm(features) > 0:
        features = features / np.linalg.norm(features)
    
    return features


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_feature_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Calculate distance between two feature vectors
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        
    Returns:
        Distance value (lower = more similar)
    """
    # Use cosine distance (1 - cosine similarity)
    similarity = cosine_similarity([features1], [features2])[0][0]
    return 1 - similarity


def calculate_center_distance(center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two centers
    
    Args:
        center1: First center point (x, y)
        center2: Second center point (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def draw_tracking_info(frame: np.ndarray, tracks: List[Any]) -> np.ndarray:
    """
    Draw tracking information on frame
    
    Args:
        frame: Input frame
        tracks: List of active tracks (KalmanBoxTracker objects)
        
    Returns:
        Frame with tracking visualization
    """
    frame_copy = frame.copy()
    
    for track in tracks:
        if hasattr(track, 'bbox') and track.bbox is not None:
            bbox = track.bbox
            track_id = track.id
            # Draw bounding box
            color = config.BOUNDING_BOX_COLORS['player']
            cv2.rectangle(frame_copy, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 
                         2)
            # Draw track ID
            label = f"ID: {track_id}"
            cv2.putText(frame_copy, 
                       label, 
                       (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       config.FONT_SCALE, 
                       color, 
                       config.FONT_THICKNESS)
            # Draw tracking path
            if hasattr(track, 'history') and len(track.history) > 1:
                history = track.history[-config.TRACK_LENGTH:]
                for i in range(1, len(history)):
                    pt1 = history[i-1]
                    pt2 = history[i]
                    cv2.line(frame_copy, pt1, pt2, config.BOUNDING_BOX_COLORS['track'], 2)
    
    return frame_copy


def save_tracking_results(tracking_data: List[Dict[str, Any]], output_path: str):
    """
    Save tracking results to JSON file
    
    Args:
        tracking_data: List of frame data with tracking information
        output_path: Path to save the JSON file
    """
    import json
    
    # Save the complete tracking data
    with open(output_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)


def create_output_directory(output_dir: str):
    """
    Create output directory if it doesn't exist
    
    Args:
        output_dir: Path to output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True) 