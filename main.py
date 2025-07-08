"""
Main execution script for Player Re-Identification System
"""

import cv2
import argparse
import os
import time
from tqdm import tqdm
import json
from typing import Dict, Any

import config
from detection import PlayerDetector
from player_tracker import PlayerTracker
import utils


def process_video(input_path: str, model_path: str, output_dir: str, 
                 save_video: bool = True, save_data: bool = True) -> Dict[str, Any]:
    """
    Process video for player re-identification
    
    Args:
        input_path: Path to input video
        model_path: Path to YOLOv11 model
        output_dir: Output directory
        save_video: Whether to save output video
        save_data: Whether to save tracking data
        
    Returns:
        Dictionary with processing results and metrics
    """
    print(f"Processing video: {input_path}")
    print(f"Using model: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    utils.create_output_directory(output_dir)
    
    # Initialize components
    detector = PlayerDetector(model_path)
    tracker = PlayerTracker()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer
    video_writer = None
    if save_video:
        output_video_path = os.path.join(output_dir, "tracked_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, config.OUTPUT_FPS, (width, height))
    
    # Processing variables
    frame_count = 0
    total_detections = 0
    total_tracks = 0
    processing_times = []
    tracking_data = []
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Detect players and ball
        detections = detector.detect(frame)
        detections = detector.filter_detections(detections)
        
        # Update tracker
        tracks = tracker.update(detections, frame)
        
        # Draw results
        output_frame = frame.copy()
        
        # Draw detections
        output_frame = detector.draw_detections(output_frame, detections)
        
        # Draw tracking information
        output_frame = utils.draw_tracking_info(output_frame, tracks)
        
        # Add frame information
        cv2.putText(output_frame, 
                   f"Frame: {frame_count} | Players: {len(tracks)} | Detections: {len(detections)}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, 
                   (255, 255, 255), 
                   2)
        
        # Write frame to video
        if video_writer is not None:
            video_writer.write(output_frame)
        
        # Store tracking data
        if save_data:
            frame_data = {
                'frame': frame_count,
                'detections': len(detections),
                'tracks': len(tracks),
                'track_info': tracker.get_track_info()
            }
            tracking_data.append(frame_data)
        
        # Update metrics
        total_detections += len(detections)
        total_tracks = max(total_tracks, len(tracks))
        processing_times.append(time.time() - start_time)
        
        frame_count += 1
        pbar.update(1)
        
        # Skip frames if configured
        if config.FRAME_SKIP > 1:
            for _ in range(config.FRAME_SKIP - 1):
                cap.read()
                frame_count += 1
                pbar.update(1)
    
    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    pbar.close()
    
    # Save tracking data
    if save_data:
        output_data_path = os.path.join(output_dir, "tracking_results.json")
        utils.save_tracking_results(tracking_data, output_data_path)
    
    # Calculate metrics
    avg_processing_time = sum(processing_times) / len(processing_times)
    avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
    
    results = {
        'input_video': input_path,
        'output_directory': output_dir,
        'total_frames': frame_count,
        'total_detections': total_detections,
        'max_concurrent_tracks': total_tracks,
        'avg_processing_time': avg_processing_time,
        'avg_fps': avg_fps,
        'video_saved': save_video,
        'data_saved': save_data
    }
    
    print("\nProcessing completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average processing time: {avg_processing_time:.3f}s per frame")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Maximum concurrent tracks: {total_tracks}")
    
    if save_video:
        print(f"Output video saved to: {os.path.join(output_dir, 'tracked_video.mp4')}")
    
    if save_data:
        print(f"Tracking data saved to: {os.path.join(output_dir, 'tracking_results.json')}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Player Re-Identification System")
    parser.add_argument("--input", "-i", 
                       default=config.DEFAULT_INPUT_VIDEO,
                       help="Input video path")
    parser.add_argument("--model", "-m", 
                       default=config.DEFAULT_MODEL_PATH,
                       help="YOLOv11 model path")
    parser.add_argument("--output", "-o", 
                       default=config.DEFAULT_OUTPUT_DIR,
                       help="Output directory")
    parser.add_argument("--no-video", 
                       action="store_true",
                       help="Don't save output video")
    parser.add_argument("--no-data", 
                       action="store_true",
                       help="Don't save tracking data")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    try:
        # Process video
        results = process_video(
            input_path=args.input,
            model_path=args.model,
            output_dir=args.output,
            save_video=not args.no_video,
            save_data=not args.no_data
        )
        
        # Save results summary
        results_path = os.path.join(args.output, "processing_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results summary saved to: {results_path}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 