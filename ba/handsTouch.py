from ultralytics import YOLO
import cv2
import mediapipe as mp
import random
import math
from PyQt5.QtCore import *

class chudian(QThread):

    update_frame = pyqtSignal(object, int)

    def __init__(self):
        super().__init__()
        self.modelball = YOLO('C:/Users/HP/Desktop/things/offterm/court/ba/best.pt')
        self.points = 0
        self.rank = 0
        self.generateFlag = 0
        self.cnt = 0
        self.ballPosition = []
        self.pointPosition = []
        self.xyxy = []
        self.mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        self.pose = mpPose.Pose()


    def get_point(self, frame, flag):

        w = frame.shape[0]
        h = frame.shape[1]
        self.pointPosition.clear()
        if flag:
            circle_y = random.randint(0.25 * h, 0.5 * h)
            circle_x = random.randint(0.10 * w, 0.30 * w)
        else:
            circle_y = random.randint(0.25 * h, 0.5 * h)
            circle_x = random.randint(0.70 * w, 0.90 * w)
        self.pointPosition.append(circle_x)
        self.pointPosition.append(circle_y)

    def get_ball_position(self,index):
        ball_x = 0
        ball_y = 0
        indexs = [0, 0, 0, 0]
        self.ballPosition.clear()
        if len(index):
            indexs = index[0]
            ball_x = (indexs[0] + indexs[2]) / 2
            ball_y = (indexs[1] + indexs[3]) / 2

        self.xyxy = indexs
        self.ballPosition.append(ball_x)
        self.ballPosition.append(ball_y)

    def rank_change(self, point):
        rank = 0
        if 20 < point < 50:
            rank = 1
        elif 50 < point < 100:
            rank = 2
        else:
            rank = 3
        return rank

    def run(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        self.runningflag = True
        while self.runningflag:
            # 读取帧
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # 生成触碰点的坐标
            if self.cnt == 0:
                self.get_point(frame, self.generateFlag)

            self.cnt = self.cnt + 1

            if self.cnt == 120 - self.rank*20:
                self.cnt = 0
                self.generateFlag = ~self.generateFlag

            #     得到两个模型的结果
            result0 = self.modelball(frame, conf=0.6, max_det=1)
            result1 = self.pose.process(frame)

            # 获得球的坐标
            boxes = result0[0].boxes
            index = boxes.xyxy.cpu().numpy()

            self.get_ball_position(index)

            # 获取结果帧
            resultFrame = result0[0].plot(conf=False, labels=False, boxes=False)

            # 画上得分点
            center_coordinates = (self.pointPosition[0], self.pointPosition[1])  # 圆中心点坐标
            radius = 20  # 圆圈半径
            color = (0, 0, 255)  # 红色
            thickness = 2  # 圆圈线条厚度
            cv2.circle(resultFrame, center_coordinates, radius, color, cv2.FILLED)

            # 检测是否被碰到
            # if self.check_touch(self.ballPosition[0], self.ballPosition[1], self.pointPosition[0], self.pointPosition[1], self.xyxy):
            #      self.cnt = 0
            #      self.points = self.points+1
            #      self.generateFlag = ~self.generateFlag

            # 等级分
            self.rank = self.rank_change(self.points)

            # 画手上的表示点
            if result1.pose_landmarks:
                # mpDraw.draw_landmarks(resultFrame, results_1.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(result1.pose_landmarks.landmark):
                    if id == 19 or id == 20:
                        h, w, c = frame.shape
                        # cx and cy represent the coordinate of two indexs
                        hand_x, hand_y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(resultFrame, (hand_x, hand_y), 5, (255, 0, 0), cv2.FILLED)
                        if math.sqrt(math.pow((hand_x - self.pointPosition[0]), 2) + math.pow((hand_y - self.pointPosition[1]), 2)) < 40:
                            self.cnt = 0
                            self.points = self.points + 1
                            self.generateFlag = ~self.generateFlag
                            # print(self.cnt)
                            # print(self.points)

            # 等级分
            self.rank = self.rank_change(self.points)

            # cv2.imshow("YOLOv8 Inference", resultFrame)
            self.update_frame.emit(resultFrame, self.points)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # else:
            #     # Break the loop if there is an issue with video capture
            #     break

                # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.runningflag = False