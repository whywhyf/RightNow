import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:/Users/HP/Desktop/things/offterm/court/runs/detect/train4/weights/best.pt')
model1 = YOLO('yolov8m-pose.pt')
print('111')

# Open the video file

# Open the video file
video_path = "C:/Users/HP/Desktop/things/offterm/court/video/2.mp4"
cap = cv2.VideoCapture(video_path)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        results1 = model1(frame)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        annotated_frame1 = results1[0].plot() + results[0].plot() - frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame1)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()