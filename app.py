import os
from flask import Flask, request, render_template, url_for
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load the custom-trained YOLOv8 model
model_path = 'models/best.pt'  # Path to your custom-trained model weights
model = YOLO(model_path)

# Define class names
class_names = [
    "energydrinkbottle", "energydrinkcan", "icetea", "juice", "milkbottle", 
    "softdrinkbottle", "softdrinkcans", "softdrinkpackage", "waterbottle", 
    "waterbottlepackage", "watergallon"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image_path = None
    annotated_image_path = None

    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        # Save the uploaded file
        file_path = os.path.join('static', 'uploads', file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        print(f"Original file saved at: {file_path}")

        # Run object detection
        results = model(file_path)

        # Process the model's output
        boxes, labels, scores = process_results(results)

        # Render the annotated image
        annotated_image_path = render_annotated_image(file_path, boxes, labels, scores)
        original_image_path = os.path.basename(file_path)

        print(f"Annotated image saved at: {annotated_image_path}")

    return render_template('index.html', original_image_path=original_image_path, annotated_image_path=annotated_image_path)

def process_results(results):
    # Extract bounding boxes, labels, and scores from the YOLOv8 model's output
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Adjust based on the actual output structure
    labels_indices = results[0].boxes.cls.cpu().numpy()  # Class indices
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    labels = [class_names[int(label_idx)] for label_idx in labels_indices]  # Map indices to class names
    return boxes, labels, scores

def render_annotated_image(image_path, boxes, labels, scores):
    # Load the original image
    img = cv2.imread(image_path)

    # Annotate the image with bounding boxes, labels, and scores
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Display only high-confidence results
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Save the annotated image
    annotated_image_path = os.path.join('static', 'uploads', 'annotated_' + os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, img)
    return os.path.basename(annotated_image_path)

if __name__ == '__main__':
    app.run(debug=True)
