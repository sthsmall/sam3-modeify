#!/usr/bin/env python3
"""
Run SAM3 on VISO dataset and generate tracking results in MOT format.
"""

import os
import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoSemanticPredictor

# Configuration
DATASET_ROOT = r"d:\projects\specific\torch-learning\sam3\dataset\laboratory_data\laboratory_data"
OUTPUT_DIR = r"d:\projects\specific\torch-learning\sam3\eval\results"
VIDEO_OUTPUT_DIR = r"d:\projects\specific\torch-learning\sam3\eval\videos"
MODEL_PATH = r"d:\projects\specific\torch-learning\sam3\sam3.pt"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

def load_seqinfo(seq_path):
    """Load sequence information from seqinfo.ini"""
    seqinfo_path = os.path.join(seq_path, "seqinfo.ini")
    if not os.path.exists(seqinfo_path):
        # Create a simple seqinfo.ini if it doesn't exist
        img_dir = os.path.join(seq_path, "img1")
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        if image_files:
            first_img = cv2.imread(os.path.join(img_dir, image_files[0]))
            height, width = first_img.shape[:2]
        else:
            height, width = 1080, 1920
        
        seqinfo = {
            'name': os.path.basename(seq_path),
            'imDir': 'img1',
            'frameRate': 10,
            'seqLength': len(image_files),
            'imWidth': width,
            'imHeight': height,
            'imExt': '.jpg'
        }
        return seqinfo
    
    with open(seqinfo_path, 'r') as f:
        lines = f.readlines()
    
    seqinfo = {}
    for line in lines:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            seqinfo[key] = value
    
    return {
        'name': seqinfo.get('name', ''),
        'imDir': seqinfo.get('imDir', 'img1'),
        'frameRate': int(seqinfo.get('frameRate', 10)),
        'seqLength': int(seqinfo.get('seqLength', 0)),
        'imWidth': int(seqinfo.get('imWidth', 1024)),
        'imHeight': int(seqinfo.get('imHeight', 1024)),
        'imExt': seqinfo.get('imExt', '.jpg')
    }

def get_prompt_from_sequence(seq_name):
    """Get prompt from sequence name"""
    if seq_name == 'AIR-aircraft':
        return "airplane"
    elif seq_name == 'SAT-ship':
        return "ship"
    elif seq_name == 'AIR-ship':
        return "ship"
    elif seq_name == 'SAT-airplane':
        return "airplane"
    return "person"

def get_color_for_id(track_id):
    """Generate a consistent color for each track ID"""
    np.random.seed(track_id * 37)
    color = tuple(map(int, np.random.randint(0, 255, 3)))
    return color

def draw_tracking_results(img_dir, image_files, results, output_video_path, fps):
    """Draw tracking results with bounding boxes and trajectories on video"""
    # Read first frame to get dimensions
    first_frame_path = os.path.join(img_dir, image_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Organize results by frame
    frame_results = {}
    for result in results:
        frame_id = int(result[0])
        if frame_id not in frame_results:
            frame_results[frame_id] = []
        frame_results[frame_id].append(result)
    
    # Store trajectory history for each track
    trajectories = {}
    
    # Process each frame
    for frame_idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue
        
        # Draw results for this frame
        if frame_idx in frame_results:
            for result in frame_results[frame_idx]:
                track_id = int(result[1])
                x, y, w, h = result[2:6]
                conf = result[6]
                
                # Get color for this track ID
                color = get_color_for_id(track_id)
                
                # Update trajectory
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                
                if track_id not in trajectories:
                    trajectories[track_id] = []
                trajectories[track_id].append((center_x, center_y))
                
                # Draw trajectory
                if len(trajectories[track_id]) > 1:
                    pts = np.array(trajectories[track_id], dtype=np.int32)
                    cv2.polylines(frame, [pts], False, color, 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                
                # Draw track ID and confidence
                label = f"ID:{track_id} {conf:.2f}"
                cv2.putText(frame, label, (int(x), int(y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Draw frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to video
        out.write(frame)
    
    out.release()
    print(f"Tracking video saved to {output_video_path}")

def run_sam3_on_sequence(seq_path, output_file, prompt, seq_type_name=None, generate_video=True):
    """Run SAM3 on a sequence and generate MOT format results"""
    # Load sequence info
    seqinfo = load_seqinfo(seq_path)
    img_dir = os.path.join(seq_path, seqinfo['imDir'])
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(seqinfo['imExt'])])
    
    if not image_files:
        print(f"No image files found in {img_dir}")
        return
    
    
    
    print(f"Processing sequence {seqinfo['name']} with {len(image_files)} ")
    print(f"Using prompt: {prompt}")
    
    # Store results in MOT format
    results = []
    
    # Create a video from the image sequence for tracking
    temp_video_path = os.path.join(OUTPUT_DIR, f"temp_{seqinfo['name']}.mp4")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(img_dir, image_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"Failed to read first frame: {image_files[0]}")
        return
    
    height, width = first_frame.shape[:2]
    fps = seqinfo['frameRate']
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # Write all frames to temporary video
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)
    out.release()
    
    print(f"Created temporary video: {temp_video_path}")
    
    # Initialize SAM3 video predictor
    overrides = dict(
        conf=0.5, 
        task="segment", 
        mode="predict", 
        model=MODEL_PATH, 
        half=True, 
    )
    
    # Use SAM3VideoSemanticPredictor for tracking
    predictor = SAM3VideoSemanticPredictor(
        overrides=overrides,
        score_threshold_detection=0.25,
        max_trk_keep_alive=100,
        init_trk_keep_alive=100,
        assoc_iou_thresh=0.1,
        trk_assoc_iou_thresh=0.1,
        new_det_thresh=0.3,
        det_nms_thresh=0.3,
        hotstart_delay = 10
    )
    
    # Run tracking on the temporary video with text prompt
    results_iter = predictor(source=temp_video_path, text=[prompt], stream=True)
    
    # Process results
    for frame_idx, result in enumerate(results_iter, 1):
        # Process each detected object
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, box in enumerate(result.boxes):
                # Get box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                conf = box.conf[0].cpu().numpy()
                
                # Get track ID if available
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())
                else:
                    continue  # Skip if no track ID available
                
                # MOT format: frame_id, id, x, y, w, h, conf, -1, -1, -1
                results.append([
                    frame_idx,
                    track_id,
                    x1,
                    y1,
                    w,
                    h,
                    conf,
                    -1,
                    -1,
                    -1
                ])
        
        if frame_idx % 5 == 0:
            print(f"Processed frame {frame_idx}/{len(image_files)}")
        

    
    # Write results to file
    with open(output_file, 'w') as f:
        # Write tracking results
        for line in results:
            line_str = ','.join(map(str, line))
            f.write(line_str + '\n')
    
    print(f"Results written to {output_file}")
    
    # Generate tracking video
    if generate_video and results:
        # Create type-specific video directory
        type_video_dir = os.path.join(VIDEO_OUTPUT_DIR, seq_type_name)
        os.makedirs(type_video_dir, exist_ok=True)
        
        video_name = f"{seqinfo['name']}_tracking.mp4"
        video_output_path = os.path.join(type_video_dir, video_name)
        draw_tracking_results(img_dir, image_files, results, video_output_path, fps)
    
    # Clean up temporary video
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        print(f"Removed temporary video: {temp_video_path}")

def main():
    """Main function"""
    # Save parameters to a single file
    params_file = os.path.join(OUTPUT_DIR, 'parameters.txt')
    with open(params_file, 'w') as f:
        f.write("# SAM3 Tracking Parameters\n")
        f.write("# Date: 2026-03-26\n")
        f.write("#\n")
        f.write("# Model parameters\n")
        f.write("conf: 0.25\n")
        f.write("task: segment\n")
        f.write(f"model: {MODEL_PATH}\n")
        f.write("half: True\n")
        f.write("#\n")
        f.write("# Tracking parameters\n")
        f.write("score_threshold_detection: 0.25\n")
        f.write("max_trk_keep_alive: 100\n")
        f.write("init_trk_keep_alive: 100\n")
        f.write("assoc_iou_thresh: 0.1\n")
        f.write("trk_assoc_iou_thresh: 0.1\n")
        f.write("new_det_thresh: 0.3\n")
        f.write("det_nms_thresh: 0.3\n")
        f.write("hotstart_delay: 10\n")
        f.write("#\n")
        f.write("# Dataset parameters\n")
        f.write(f"dataset_root: {DATASET_ROOT}\n")
        f.write(f"output_dir: {OUTPUT_DIR}\n")
        f.write(f"video_output_dir: {VIDEO_OUTPUT_DIR}\n")
    
    print(f"Parameters saved to {params_file}")
    
    # Process AIR- and SAT- sequences
    # sequence_types = ['AIR-aircraft', 'SAT-ship', 'AIR-ship', 'SAT-airplane']
    sequence_types = ['AIR-aircraft', 'SAT-ship', 'AIR-ship', 'SAT-airplane']
    
    for seq_type in sequence_types:
        seq_type_path = os.path.join(DATASET_ROOT, seq_type)
        if not os.path.exists(seq_type_path):
            print(f"Sequence type {seq_type} not found")
            continue
        
        # Get train directory
        train_path = os.path.join(seq_type_path, 'train')
        if not os.path.exists(train_path):
            print(f"Train directory not found for {seq_type}")
            continue
        
        # Get all sequence directories and filter out invalid ones
        sequences = []
        for d in os.listdir(train_path):
            seq_path = os.path.join(train_path, d)
            if os.path.isdir(seq_path) and os.path.exists(os.path.join(seq_path, "img1")):
                sequences.append(d)
        
        total_sequences = len(sequences)
        if not sequences:
            print(f"No sequences found for {seq_type}")
            continue
        
        # Create type-specific results directory
        type_results_dir = os.path.join(OUTPUT_DIR, seq_type)
        os.makedirs(type_results_dir, exist_ok=True)
        
        # Run all valid sequences
        for i, seq in enumerate(sequences):
            seq_path = os.path.join(train_path, seq)
            output_file = os.path.join(type_results_dir, f"{seq}.txt")
            
            # Get prompt from sequence type
            prompt = get_prompt_from_sequence(seq_type)
            
            print(f"Processing {seq_type} - {seq} ({i+1}/{total_sequences})")
            # Run SAM3 on this sequence
            run_sam3_on_sequence(seq_path, output_file, prompt, seq_type_name=seq_type, generate_video=True)
            print(f"Completed processing {seq_type} - {seq} ({i+1}/{total_sequences})")

if __name__ == "__main__":
    main()
