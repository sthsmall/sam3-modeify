import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor
model_path = r"d:\projects\specific\torch-learning\sam3\sam3.pt"
overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, half=True, save=True)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Load image
image_path = r"d:\projects\specific\torch-learning\sam3\000018.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Failed to load image from {image_path}")
    exit(1)

print("Loading image into SAM3 model...")
predictor.set_image(image_path)

# Interactive bounding box selection
print("\n=== Interactive Bounding Box Selection ===")
print("Drag to draw a rectangle, press ENTER to confirm")
print("Press 'c' to cancel current selection")
print("Press 'd' to finish selection and process all boxes")
print("Press 'q' to quit without processing")

bboxes = []

def draw_bboxes(img, boxes):
    for i, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"Box {i+1}", (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

while True:
    # Create a copy for selection and draw existing bboxes
    image_copy = image.copy()
    image_copy = draw_bboxes(image_copy, bboxes)
    
    # Select ROI (Region of Interest)
    bbox = cv2.selectROI("Select Object to Segment", image_copy, fromCenter=False, showCrosshair=True)
    
    # If user pressed ESC or closed window, bbox will be (0,0,0,0)
    if bbox == (0, 0, 0, 0):
        print("Selection cancelled or window closed.")
        break
    
    x, y, w, h = bbox
    print(f"\nSelected bbox: x={x}, y={y}, w={w}, h={h}")
    
    # Convert to [x1, y1, x2, y2] format for SAM3
    bbox_formatted = [x, y, x + w, y + h]
    bboxes.append(bbox_formatted)
    print(f"Added to list. Total bboxes: {len(bboxes)}")
    print(f"Current bboxes: {bboxes}")
    
    # Ask for next action
    print("\nPress 'd' to process all selected boxes")
    print("Press any other key to continue selecting another region")
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('d'):
        if len(bboxes) == 0:
            print("No boxes selected. Please select at least one box.")
            continue
        print(f"\nProcessing {len(bboxes)} boxes...")
        
        # Perform segmentation with all bboxes
        results = predictor(bboxes=bboxes)
        
        # Display results
        if results and len(results) > 0:
            result_img = results[0].plot()
            cv2.imshow("Segmentation Result", result_img)
            print(f"Saved result to: {results[0].save_dir}")
            print(f"Results: {results[0]}")
            
            print("\nPress any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow("Segmentation Result")
        
        # Reset for next batch
        bboxes = []
        print("\nReady for next batch of selections...")
    elif key == ord('q'):
        print("Exiting...")
        break

cv2.destroyAllWindows()
print("Done!")
