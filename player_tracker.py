"""
Player Tracking and Re-Identification Module
Implements SORT (Simple Online and Realtime Tracking) with re-identification
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import config
import utils


class KalmanBoxTracker:
    """
    Kalman Filter tracker for individual player tracking
    """
    
    def __init__(self, bbox: List[int], track_id: int):
        """
        Initialize tracker with initial bounding box
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            track_id: Unique track ID
        """
        self.id = track_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.history = [self.center]
        self.features = []
        
        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State: [u, v, s, r, u', v', s']
        # u, v: center coordinates
        # s: scale (area)
        # r: aspect ratio
        # u', v', s': velocities
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        
        # Initialize state
        u, v = self.center
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        self.kf.x[:4, 0] = [u, v, s, r]
    
    def predict(self):
        """Predict next state using Kalman Filter"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, bbox: List[int], features: Optional[np.ndarray] = None):
        """
        Update tracker with new detection
        
        Args:
            bbox: New bounding box
            features: Appearance features (optional)
        """
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.history.append(self.center)
        
        # Keep only recent history
        if len(self.history) > config.TRACK_LENGTH:
            self.history = self.history[-config.TRACK_LENGTH:]
        
        # Update Kalman Filter
        u, v = self.center
        s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        r = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        self.kf.update([u, v, s, r])
        
        self.hits += 1
        self.time_since_update = 0
        
        # Store features for re-identification
        if features is not None:
            self.features.append(features)
            if len(self.features) > config.REID_HISTORY_SIZE:
                self.features = self.features[-config.REID_HISTORY_SIZE:]
    
    def get_state(self) -> List[int]:
        """Get current bounding box state"""
        state = self.kf.x[:4]
        u, v, s, r = state
        
        if s <= 0 or r <= 0:
            # Return last known bbox if invalid
            return self.bbox
        w = np.sqrt(s * r)
        h = s / w if w != 0 else 0
        if w <= 0 or h <= 0:
            return self.bbox
        x1 = int(u - w / 2)
        y1 = int(v - h / 2)
        x2 = int(u + w / 2)
        y2 = int(v + h / 2)
        return [x1, y1, x2, y2]


class PlayerTracker:
    """
    Main player tracking system with re-identification
    """
    
    def __init__(self):
        """Initialize the tracker"""
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
        self.lost_tracks = []  # Tracks that went out of view
    
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from YOLOv11
            frame: Current frame for feature extraction
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Filter player detections
        player_detections = [det for det in detections if det['object_type'] == 'player']
        
        # Extract features for all detections
        detection_features = []
        for det in player_detections:
            features = utils.extract_appearance_features(frame, det['bbox'])
            detection_features.append(features)
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to existing tracks
        matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = \
            self._associate_detections_to_tracks(player_detections, detection_features)
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            track = self.tracks[track_idx]
            det = player_detections[det_idx]
            features = detection_features[det_idx]
            track.update(det['bbox'], features)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = player_detections[det_idx]
            features = detection_features[det_idx]
            
            # Try to re-identify with lost tracks
            reidentified_track = self._reidentify_lost_track(det, features)
            
            if reidentified_track is not None:
                # Re-identify lost track
                reidentified_track.update(det['bbox'], features)
                self.tracks.append(reidentified_track)
                self.lost_tracks.remove(reidentified_track)
            else:
                # Create new track
                new_track = KalmanBoxTracker(det['bbox'], self.next_id)
                new_track.update(det['bbox'], features)
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update < config.TRACKING_MAX_AGE]
        
        # Move tracks to lost_tracks if they haven't been updated
        tracks_to_remove = []
        for track in self.tracks:
            if track.time_since_update >= config.TRACKING_MAX_AGE and track.hits >= config.TRACKING_MIN_HITS:
                self.lost_tracks.append(track)
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.tracks.remove(track)
        
        # Clean up old lost tracks
        self.lost_tracks = [track for track in self.lost_tracks 
                           if track.time_since_update < config.TRACKING_MAX_AGE * 2]
        
        return self.tracks
    
    def _associate_detections_to_tracks(self, detections: List[Dict[str, Any]], 
                                      detection_features: List[np.ndarray]) -> Tuple:
        """
        Associate detections to existing tracks using Hungarian algorithm
        
        Returns:
            Tuple of (matched_tracks, matched_detections, unmatched_tracks, unmatched_detections)
        """
        if len(self.tracks) == 0:
            return [], [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks))), []
        
        # Calculate cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            track_state = track.get_state()
            
            for j, det in enumerate(detections):
                # Calculate IoU cost
                iou_cost = 1 - utils.calculate_iou(track_state, det['bbox'])
                
                # Calculate center distance cost
                center_dist = utils.calculate_center_distance(track.center, det['center'])
                center_cost = min(center_dist / 100.0, 1.0)  # Normalize to [0, 1]
                
                # Calculate appearance cost if features are available
                appearance_cost = 0.0
                if len(track.features) > 0 and len(detection_features) > 0:
                    # Use the most recent feature
                    track_feature = track.features[-1]
                    det_feature = detection_features[j]
                    appearance_cost = utils.calculate_feature_distance(track_feature, det_feature)
                
                # Combined cost (weighted average)
                cost_matrix[i, j] = 0.5 * iou_cost + 0.3 * center_cost + 0.2 * appearance_cost
        
        # Apply Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on cost threshold
        matched_tracks = []
        matched_detections = []
        unmatched_tracks = []
        unmatched_detections = []
        
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, det_idx] < config.REID_DISTANCE_THRESHOLD:
                matched_tracks.append(track_idx)
                matched_detections.append(det_idx)
            else:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(det_idx)
        
        # Add unmatched tracks and detections
        for i in range(len(self.tracks)):
            if i not in matched_tracks:
                unmatched_tracks.append(i)
        
        for i in range(len(detections)):
            if i not in matched_detections:
                unmatched_detections.append(i)
        
        return matched_tracks, matched_detections, unmatched_tracks, unmatched_detections
    
    def _reidentify_lost_track(self, detection: Dict[str, Any], 
                              features: np.ndarray) -> Optional[KalmanBoxTracker]:
        """
        Try to re-identify a lost track using appearance features
        
        Args:
            detection: New detection
            features: Appearance features of the detection
            
        Returns:
            Re-identified track or None
        """
        if len(self.lost_tracks) == 0:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for track in self.lost_tracks:
            if len(track.features) == 0:
                continue
            
            # Calculate distance to track's features
            min_distance = float('inf')
            for track_feature in track.features:
                distance = utils.calculate_feature_distance(features, track_feature)
                min_distance = min(min_distance, distance)
            
            # Also consider spatial proximity
            center_dist = utils.calculate_center_distance(track.center, detection['center'])
            spatial_cost = min(center_dist / 200.0, 1.0)  # Normalize
            
            # Combined distance
            combined_distance = 0.7 * min_distance + 0.3 * spatial_cost
            
            if combined_distance < best_distance and combined_distance < config.REID_DISTANCE_THRESHOLD:
                best_distance = combined_distance
                best_match = track
        
        return best_match
    
    def get_track_info(self) -> List[Dict[str, Any]]:
        """Get information about all active tracks"""
        track_info = []
        for track in self.tracks:
            info = {
                'id': track.id,
                'bbox': track.bbox,
                'center': track.center,
                'age': track.age,
                'hits': track.hits,
                'time_since_update': track.time_since_update,
                'history': track.history.copy()
            }
            track_info.append(info)
        return track_info 