import cv2
from ultralytics import YOLO
import time
# Load the YOLOv8 model
model = YOLO('C:/Users/HP/Desktop/things/offterm/court/train/runs/detect/train9/weights/best.pt')
# model = YOLO('test2.pt')

print('111')
# Open the video file
video_path = "./video/4.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # time.sleep(0.05)
        # Run YOLOv8 inference on the frame
        results = model(frame)
        # probs = results[0].probs  # cls prob, (num_class, )
        # probs.top5    # The top5 indices of classification, List[Int] * 5.
        # probs.top1    # The top1 indices of classification, a value with Int type.
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # probs = results[0][0].verbose()
        # print("probs:", probs)
        # print('classes:',len(results[0]))
        for i in range(len(results[0])):
            print('num:',i,results[0][i].verbose())
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                print("here:",model.names[box.cls[0]])
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()