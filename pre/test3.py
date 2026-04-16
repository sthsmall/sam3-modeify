import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

model_path = r"d:\projects\specific\torch-learning\sam3\sam3.pt"
overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, half=True, save=True)
predictor = SAM3SemanticPredictor(overrides=overrides)

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

output_path = r"d:\projects\specific\torch-learning\sam3\video_segmentation_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames}")
    
    predictor.set_image(frame)
    result = predictor(text=["plane"])
    
    if result and len(result) > 0:
        result_img = result[0].plot()
        out.write(result_img)
    else:
        out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video segmentation completed!")
print(f"Result saved to: {output_path}")
