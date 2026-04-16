import cv2
import numpy as np
from ultralytics.models.sam import SAM3VideoSemanticPredictor

model_path = r"d:\projects\specific\torch-learning\sam3\sam3.pt"
overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, half=True, compile=None)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

video_path = r"d:\projects\specific\torch-learning\sam3\runs\segment\predict\3月20日1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

cap.release()

print("\n=== Auto Track All Objects ===")
print("Using text prompt 'plane' to detect and track all objects")

output_path = r"d:\projects\specific\torch-learning\sam3\video_auto_track_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"\nProcessing video with SAM3VideoSemanticPredictor...")

# results = predictor(source=video_path, text=["plane"], stream=True)
results = predictor(source=video_path, text=["ship"], stream=True)

frame_count = 0
for r in results:
    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames}")
    
    result_img = r.plot()
    out.write(result_img)

out.release()
cv2.destroyAllWindows()

print(f"\nVideo auto tracking completed!")
print(f"Result saved to: {output_path}")
