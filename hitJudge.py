import cv2
from ultralytics import YOLO
import time
from function import calculateAngle, getAngleFromDatum, getSpeedFromDatum, getAverage


# Load the YOLOv8 model
# 25代 s模型
model = YOLO('./train/runs/detect/train12/weights/best.pt')
# 60代 n模型
# model = YOLO('test5.pt')
# yolo姿势模型
modelpose = YOLO('yolov8n-pose.pt')
# model = YOLO('yolov8n-pose.pt')

print('111')
# Open the video file
video_path = "./video/5.mp4"
cap = cv2.VideoCapture(video_path)
# 获取视频帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象，用于写入 AVI 文件
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编解码器
output_video = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

# 球的轨迹
trace = [[-1, -1]]

# 出手数
attempts = 0
attemptFLAG = True

# 命中数
hit = 0

# 初始化篮球 球筐位置
ballLast = [-1, -1]
basketLast = [-1, -1, -1, -1]


# standard time
timescale = 0.05
# shooting time
shoot_time = 0


# Whether we shoot
initial_flag = 0
shoot_flag = 0
# numpy
shoot_num = []
elbowAngle_num = []
kneeAngle_num = []
# average data
average_shoot = 0
average_elbowAngle = 0
average_kneeAngle = 0

# 获取篮球
def getBall(ballLast, boxes):
    ballxy = ballLast
    for box in boxes:
        if box.cls[0] == 1:
            if ballLast[0] == -1 or abs(box.xyxy[0][0] - ballLast[0]) + abs(box.xyxy[0][1] - ballLast[1]) < 200:
                # print('distance:',abs(box.xyxy[0][0] - ballLast[0]) + abs(box.xyxy[0][1] - ballLast[1]), ballLast[0])
                # print(box)
                ballxy = box.xyxy[0]
                ballLast = ballxy
                print('BALLXY:',ballxy)
                break
    return ballxy, ballLast

# 获取球筐
def getBasket(basketLast, boxes):
    basketxy = basketLast
    for box in boxes:
        if box.cls[0] == 0:
            basketxy = box.xyxy[0]
            basketLast = basketxy
            print('BASKETXY:',basketxy)
            break
    return basketxy, basketLast

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()


    if success:
        # time.sleep(0.05)
        # Run YOLOv8 inference on the frame
        results = model(frame)
        resultspose = modelpose(frame)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        annotated_frame = results[0].plot()
        annotated_framepose = resultspose[0].plot()
        annotated_frame_out = annotated_framepose + annotated_frame - frame




        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                print("here:",model.names[box.cls[0]])





        boxes = results[0].boxes.cpu().numpy()
        # 获取篮球
        # ballxy = ballLast
        # for box in boxes:
        #     if box.cls[0] == 1:
        #         if ballLast[0] == -1 or abs(box.xyxy[0][0] - ballLast[0]) + abs(box.xyxy[0][1] - ballLast[1]) < 200:
        #             # print(box)
        #             ballxy = box.xyxy[0]
        #             ballLast = ballxy
        #             print('BALLXY:',ballxy)
        #             break
        ballxy, ballLast = getBall(ballLast, boxes)

        
        # 获取篮筐
        # basketxy = basketLast
        # for box in boxes:
        #     if box.cls[0] == 0:
        #         basketxy = box.xyxy[0]
        #         basketLast = basketxy
        #         print('BASKETXY:',basketxy)
        #         break
        basketxy, basketLast = getBasket(basketLast, boxes)





        # 若识别出球，将坐标加入trace（没检测也插入空帧，则可以确定时间间隔），并显示球的位置
        if ballxy[0] != -1:
            # print('error:',ballxy)
            trace.append([(ballxy[0] + ballxy[2]) / 2, (ballxy[1] + ballxy[3]) / 2])  
            cv2.circle(annotated_frame_out,(int((ballxy[0] + ballxy[2]) / 2), int((ballxy[1] + ballxy[3]) / 2)),5,(0,0,255),-1) 
        else:
            trace.append([-1, -1])  

        # 获取球筐的边界
        if basketxy[0] != -1:
            cv2.rectangle(annotated_frame_out, (int(basketxy[0]+5), int(basketxy[1])), (int(basketxy[2]-5), int(basketxy[3]+ 50)), (0, 0, 255), 2)
            # cv2.rectangle(annotated_frame, (50, 100), (300, 200), (0, 0, 255), 2)
        # if basketxy[0] != -1:
        #     basketHigh = basketxy[1]
        #     basketLow = basketxy[3]


        # 判断是否出手
        if ballxy != [-1, -1]:
            print('FLAG:',attemptFLAG)
            print("BALL:",ballxy[0],ballxy[1])
            if attemptFLAG == True and ballxy[1] < basketxy[1] + 5:# 若高于篮筐
                print('AT:',ballxy[0], ballxy[1])
                attemptFLAG = False
                attempts += 1
            elif attemptFLAG == False and ballxy[1] > basketxy[3] + 50:# 若低于篮筐较多
               attemptFLAG = True
        



        # 判定是否进球
        end = len(trace) - 1
        
        if ballxy[0] > basketxy[0]  and ballxy[0] < basketxy[2] and ballxy[1] > basketxy[1] + 5 and ballxy[1] < basketxy[3] + 50:
            while trace[end - 1] == [-1, -1] and end > 1:
                end -= 1
            ballPre = trace[end - 1]
            # hit += 1
            print('last:',ballLast)
            if ballPre[0] >  basketxy[0] - 100 and ballPre[0] < basketxy[2] + 100  and ballPre[1] < basketxy[1]:
                hit += 1


        # 显示进球数
        cv2.putText(annotated_frame_out, "ATTEMPTS:"+str(attempts), (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(annotated_frame_out, "HITS:"+str(hit), (1, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        # print(results[0])
        # print(annotated_frame)
        # Display the annotated frame

        # calculate angle
        elbowAngle, kneeAngle, elbowCoord, kneeCoord = getAngleFromDatum(resultspose[0])
        if elbowAngle != 0:
            cv2.putText(annotated_frame_out, 'Elbow: ' + str(elbowAngle) + ' deg',
                        (elbowCoord[0] + 65, elbowCoord[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (102, 255, 0), 1)
            cv2.putText(annotated_frame_out, 'Knee: ' + str(kneeAngle) + ' deg',
                        (kneeCoord[0] + 65, kneeCoord[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (102, 255, 0), 1)
        # calculate the shooting time
        shoot_flag, shoot_time = getSpeedFromDatum(resultspose[0], shoot_flag, shoot_time, timescale)
        cv2.putText(annotated_frame_out, 'Time' + str(shoot_time), (1, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        # calculate the avarage data
        if shoot_flag:
            if initial_flag == 0:
                elbowAngle_num.append(elbowAngle)
                kneeAngle_num.append(kneeAngle)
                shoot_num.append(shoot_time)
                average_shoot = getAverage(shoot_num)
                average_elbowAngle = getAverage(elbowAngle_num)
                average_kneeAngle = getAverage(kneeAngle_num)
        initial_flag = shoot_flag
        print(average_shoot, average_kneeAngle, average_elbowAngle)


        cv2.imshow("YOLOv8 Inference", annotated_frame_out)

        # 将当前帧写入 AVI 文件
        output_video.write(annotated_frame_out)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            output_video.release()
            cv2.destroyAllWindows()
            for i in range(len(trace)):
                print('TRACE:',trace[i])
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()