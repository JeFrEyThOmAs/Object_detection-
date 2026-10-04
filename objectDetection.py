import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 nano model (fastest for real-time webcam use)
model = YOLO('yolov8n.pt')

# 2. Access the default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Starting video stream. Press 'q' to quit.")

# 3. Create the real-time processing loop
while True:
    # Read the current frame from the webcam
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab frame.")
        break
        
    # 4. Pass the frame to the AI model for detection
    # stream=True keeps it fast and memory-efficient
    results = model(frame, stream=True)

    # 5. Draw the bounding boxes and labels on the frame
    for r in results:
        annotated_frame = r.plot()

    # 6. Display the live feed with detections
    cv2.imshow("Group 4: Real-Time Object Detection", annotated_frame)

    # 7. Listen for the 'q' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up and close the window
cap.release()
cv2.destroyAllWindows()