import os
import cv2
import insightface
from insightface.app import FaceAnalysis

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test5.jpg")
output_path = os.path.join(script_dir, "output_detected.jpg")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 1280))

img = cv2.imread("test5.jpg")
faces = app.get(img)
print(f"The total number of faces detected are: {len(faces)}")

# 5. DRAW BOUNDING BOXES & SCORES ON THE IMAGE
for idx, face in enumerate(faces):
    # Convert box float coordinates to integers
    x1, y1, x2, y2 = face.bbox.astype(int)
    confidence = face.det_score

    # Draw green box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 1. DEFINE LABEL HERE BEFORE USING IT
    label = f"#{idx + 1} ({confidence:.2f})"
    
    # 2. DRAW TEXT
    cv2.putText(
        img, 
        label, 
        (x1, max(12, y1 - 10)),  # max() prevents text from getting cut off at the top edge
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 255, 0), 
        2
    )

cv2.imwrite(output_path, img)
print(f" Annotated image saved to: {output_path}")

cv2.imshow("Detected Faces", img)
cv2.waitKey(0)  
cv2.destroyAllWindows()