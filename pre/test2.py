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

result = predictor(text=["plane"])

print(result)

# Display result using the built-in plot method
if result and len(result) > 0:
    # Use the built-in plot method to draw results on the original image
    result_img = result[0].plot()
    
    # Save the result to file
    output_path = r"d:\projects\specific\torch-learning\sam3\segmentation_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"Segmentation result saved to {output_path}")
    
    try:
        # Try to display the result
        cv2.imshow('SAM3 Segmentation Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Could not display image due to environment restrictions: {e}")
        print("The result has been saved to file instead.")
else:
    print("No results found")