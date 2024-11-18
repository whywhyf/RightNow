import cv2
import time
import numpy as np
from ultralytics import YOLO
from PyQt5.QtCore import *

# 投篮分析
class ShootThread(QThread):
    finished = pyqtSignal(int, int, float, float, float)
    update_frame = pyqtSignal(object, int, int)

    def __init__(self, url):
        super().__init__()
        # Load the YOLOv8 model
        self.model = YOLO('C:/Users/HP/Desktop/things/offterm/court/train/runs/detect/train12/weights/best.pt')
        self.modelpose = YOLO('C:/Users/HP/Desktop/things/offterm/court/yolov8n-pose.pt')

        # Open the video file
        self.video_path = url[8:]
        print(self.video_path)
        self.cap = cv2.VideoCapture(self.video_path)

        # 获取视频帧率和尺寸
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建 VideoWriter 对象，用于写入 AVI 文件
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编解码器
        self.output_video = cv2.VideoWriter('./basketball/output_video.avi', self.fourcc, self.fps, (self.width, self.height))
        # self.output_video2 = cv2.VideoWriter('./basketball/output_video2.avi', self.fourcc, self.fps, (self.width, self.height))

        # 球的轨迹
        self.trace = [[-1, -1]]

        # 出手数
        self.attempts = 0
        self.attemptFLAG = True

        # 命中数
        self.hit = 0

        # 初始化篮球 球筐位置
        self.ballLast = [-1, -1]
        self.basketLast = [-1, -1, -1, -1]


        # standard time
        self.timescale = 0.05
        # shooting time
        self.shoot_time = 0

        # Whether we shoot
        self.initial_flag = 0
        self.shoot_flag = 0
        # numpy
        self.shoot_num = []
        self.elbowAngle_num = []
        self.kneeAngle_num = []
        # average data
        self.average_shoot = 0
        self.average_elbowAngle = 0
        self.average_kneeAngle = 0
        print('look')
    
    # def run(self):
    #     print('hello')
    
    def run(self):
        # 在这里编写线程执行的代码
        # Loop through the video frames
        print('ikun')
        while self.cap.isOpened():
            print('here')
            # Read a frame from the video
            success, frame = self.cap.read()


            if success:
                print('here2')

                # time.sleep(0.05)
                # Run YOLOv8 inference on the frame
                results = self.model(frame)
                resultspose = self.modelpose(frame)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                annotated_framepose = resultspose[0].plot()
                annotated_frame_out = annotated_framepose + annotated_frame - frame




                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        print("here:",self.model.names[box.cls[0]])


                print('here3')


                boxes = results[0].boxes.cpu().numpy()
                # 获取篮球
                ballxy = self.ballLast
                for box in boxes:
                    if box.cls[0] == 1:
                        if self.ballLast[0] == -1 or abs(box.xyxy[0][0] - self.ballLast[0]) + abs(box.xyxy[0][1] - self.ballLast[1]) < 200:
                            # print(box)
                            ballxy = box.xyxy[0]
                            self.ballLast = ballxy
                            print('BALLXY:',ballxy)
                            break
                # ballxy, ballLast = getBall(ballLast, boxes)

                
                # 获取篮筐
                basketxy = self.basketLast
                for box in boxes:
                    if box.cls[0] == 0:
                        basketxy = box.xyxy[0]
                        self.basketLast = basketxy
                        print('BASKETXY:',basketxy)
                        break
                # basketxy, basketLast = getBasket(basketLast, boxes)

                # print('here3')



                # 若识别出球，将坐标加入trace（没检测也插入空帧，则可以确定时间间隔），并显示球的位置
                if ballxy[0] != -1:
                    # print('error:',ballxy)
                    self.trace.append([(ballxy[0] + ballxy[2]) / 2, (ballxy[1] + ballxy[3]) / 2])  
                    cv2.circle(annotated_frame_out,(int((ballxy[0] + ballxy[2]) / 2), int((ballxy[1] + ballxy[3]) / 2)),5,(0,0,255),-1) 
                else:
                    self.trace.append([-1, -1])  

                # 获取球筐的边界
                if basketxy[0] != -1:
                    cv2.rectangle(annotated_frame_out, (int(basketxy[0]+5), int(basketxy[1])), (int(basketxy[2]-5), int(basketxy[3]+ 50)), (0, 0, 255), 2)
                    # cv2.rectangle(annotated_frame, (50, 100), (300, 200), (0, 0, 255), 2)
                # if basketxy[0] != -1:
                #     basketHigh = basketxy[1]
                #     basketLow = basketxy[3]


                # 判断是否出手
                if ballxy != [-1, -1]:
                    print('FLAG:',self.attemptFLAG)
                    print("BALL:",ballxy[0],ballxy[1])
                    if self.attemptFLAG == True and ballxy[1] < basketxy[1] + 5:# 若高于篮筐
                        print('AT:',ballxy[0], ballxy[1])
                        self.attemptFLAG = False
                        self.attempts += 1
                    elif self.attemptFLAG == False and ballxy[1] > basketxy[3] + 50:# 若低于篮筐较多
                        self.attemptFLAG = True
                


                
                # 判定是否进球
                end = len(self.trace) - 1
                
                if ballxy[0] > basketxy[0]  and ballxy[0] < basketxy[2] and ballxy[1] > basketxy[1] + 5 and ballxy[1] < basketxy[3] + 50:
                    while self.trace[end - 1] == [-1, -1] and end > 1:
                        end -= 1
                    ballPre = self.trace[end - 1]
                    # hit += 1
                    print('last:',self.ballLast)
                    if ballPre[0] >  basketxy[0] - 100 and ballPre[0] < basketxy[2] + 100  and ballPre[1] < basketxy[1]:
                        self.hit += 1
                

                # 显示进球数
                cv2.putText(annotated_frame_out, "ATTEMPTS:"+str(self.attempts), (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(annotated_frame_out, "HITS:"+str(self.hit), (1, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                # print(results[0])
                # print(annotated_frame)
                # Display the annotated frame

                print('here4')
                # calculate angle
                elbowAngle, kneeAngle, elbowCoord, kneeCoord = self.getAngleFromDatum(resultspose[0])
                print('here5')

                if elbowAngle != 0:
                    cv2.putText(annotated_frame_out, 'Elbow: ' + str(elbowAngle) + ' deg',
                                (elbowCoord[0] + 65, elbowCoord[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (102, 255, 0), 1)
                    cv2.putText(annotated_frame_out, 'Knee: ' + str(kneeAngle) + ' deg',
                                (kneeCoord[0] + 65, kneeCoord[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (102, 255, 0), 1)
                # calculate the shooting time
                self.shoot_flag, self.shoot_time = self.getSpeedFromDatum(resultspose[0], self.shoot_flag, self.shoot_time, self.timescale)
                cv2.putText(annotated_frame_out, 'Time' + str(self.shoot_time), (1, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
                

                
                # calculate the avarage data
                if self.shoot_flag:
                    if self.initial_flag == 0:
                        self.elbowAngle_num.append(elbowAngle)
                        self.kneeAngle_num.append(kneeAngle)
                        self.shoot_num.append(self.shoot_time)
                        self.average_shoot = self.getAverage(self.shoot_num)
                        self.average_elbowAngle = self.getAverage(self.elbowAngle_num)
                        self.average_kneeAngle = self.getAverage(self.kneeAngle_num)
                self.initial_flag = self.shoot_flag
                print(self.average_shoot, self.average_kneeAngle, self.average_elbowAngle)

                # 显示画面
                cv2.imshow("YOLOv8 Inference", annotated_frame_out)

                # 回传画面
                self.update_frame.emit(frame, self.hit, self.attempts)

                # 将当前帧写入 AVI 文件
                self.output_video.write(annotated_frame_out)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.cap.release()
                    self.output_video.release()
                    cv2.destroyAllWindows()
                    for i in range(len(self.trace)):
                        print('TRACE:',self.trace[i])
                    break
            else:
                # Break the loop if the end of the video is reached
                self.cap.release()
                self.output_video.release()
                cv2.destroyAllWindows()
                break
                pass
        self.finished.emit(self.hit, self.attempts, self.hit/self.attempts, self.average_shoot, self.average_elbowAngle)
        print(type(QUrl('./basketball/output_video.avi')), QUrl('./basketball/output_video.avi'))
        # Release the video capture object and close the display window
        self.cap.release()
        self.output_video.release()
        cv2.destroyAllWindows()


# function
    def calculateAngle(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return round(np.degrees(angle), 2)


    def getAngleFromDatum(self, datum):
        size = len(datum.keypoints.xy[0])
        # print('1')
        if size!= 0:
            # print('2')
            hipX, hipY = datum.keypoints.xy[0][12].cpu()
            kneeX, kneeY = datum.keypoints.xy[0][13].cpu()
            ankleX, ankleY = datum.keypoints.xy[0][15].cpu()

            shoulderX, shoulderY = datum.keypoints.xy[0][5].cpu()
            elbowX, elbowY = datum.keypoints.xy[0][7].cpu()
            wristX, wristY = datum.keypoints.xy[0][9].cpu()

            # test = np.array([hipX, hipY])
            # print('3')
            kneeAngle = self.calculateAngle(np.array([hipX, hipY]), np.array(
                [kneeX, kneeY]), np.array([ankleX, ankleY]))
            elbowAngle = self.calculateAngle(np.array([shoulderX, shoulderY]), np.array(
                [elbowX, elbowY]), np.array([wristX, wristY]))

            elbowCoord = np.array([int(elbowX), int(elbowY)])
            kneeCoord = np.array([int(kneeX), int(kneeY)])
            # print('4')
            return elbowAngle, kneeAngle, elbowCoord, kneeCoord
        else:
            return 0,0,0,0

    def getSpeedFromDatum(self, datum, status, time, timescale):
        size = len(datum.keypoints.xy[0])
        if size != 0:
            earX, earY = datum.keypoints.xy[0][4]
            elbowX, elbowY = datum.keypoints.xy[0][8]
            wristX, wristY = datum.keypoints.xy[0][10]
            if status == 1 and wristY > elbowY:
                time = 0
                status = 0
            if wristY < earY-10 and status == 0:
                status = 1
            elif wristY < elbowY and status == 0 :
                time = time + timescale
            #    shoot_time = time
            return status, time
        else:
            return 0,0

    def getAverage(self, x):
        num = 0
        total = 0
        for i in range(len(x)):
            num += 1
            total += x[i]
        average = total/num
        return  average
