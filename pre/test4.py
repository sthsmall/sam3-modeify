import cv2
import numpy as np
from ultralytics.models.sam import SAM3VideoSemanticPredictor

model_path = r"d:\projects\specific\torch-learning\sam3\sam3.pt"
overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, half=True, compile=None)

predictor = SAM3VideoSemanticPredictor(
    overrides=overrides,
    score_threshold_detection=0.3,
    max_trk_keep_alive=60,
    init_trk_keep_alive=60,
    new_det_thresh=0.0,
    masklet_confirmation_enable=False,
)

video_path = r"d:\projects\specific\torch-learning\sam3\runs\segment\predict\3月20日.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

ret, first_frame = cap.read()
if not ret:
    print("Failed to read first frame")
    exit(1)

cap.release()

print("\n=== Interactive Reference Selection ===")
print("Select reference objects to track")
print("Drag to draw a rectangle, press ENTER to confirm")
print("Press 'd' to finish selection and start tracking")
print("Press 'q' to quit without processing")

bboxes = []

def draw_bboxes(img, boxes):
    for i, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"Ref {i+1}", (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

while True:
    frame_copy = first_frame.copy()
    frame_copy = draw_bboxes(frame_copy, bboxes)
    
    bbox = cv2.selectROI("Select Reference Object", frame_copy, fromCenter=False, showCrosshair=True)
    
    if bbox == (0, 0, 0, 0):
        print("Selection cancelled or window closed.")
        break
    
    x, y, w, h = bbox
    print(f"\nSelected reference bbox: x={x}, y={y}, w={w}, h={h}")
    
    bbox_formatted = [x, y, x + w, y + h]
    bboxes.append(bbox_formatted)
    print(f"Added to list. Total references: {len(bboxes)}")
    print(f"Current bboxes: {bboxes}")
    
    print("\nPress 'd' to start tracking")
    print("Press any other key to add another reference")
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('d'):
        if len(bboxes) == 0:
            print("No boxes selected. Please select at least one box.")
            continue
        print(f"\nStarting video tracking with {len(bboxes)} reference boxes...")
        break
    elif key == ord('q'):
        print("Exiting...")
        cv2.destroyAllWindows()
        exit(0)

cv2.destroyWindow("Select Reference Object")

output_path = r"d:\projects\specific\torch-learning\sam3\video_tracking_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"\nProcessing video with SAM3VideoSemanticPredictor...")
print("Using reference bboxes to track objects")
print(f"Tracking parameters: score_threshold=0.3, keep_alive=60 frames")

results = predictor(source=video_path, bboxes=bboxes, stream=True)

frame_count = 0
for r in results:
    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames}")
    
    result_img = r.plot()
    out.write(result_img)

out.release()
cv2.destroyAllWindows()

print(f"\nVideo tracking completed!")
print(f"Result saved to: {output_path}")
