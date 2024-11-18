import sys
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from ShootThread import ShootThread
from Ui_basketball import Ui_Basketball
from HandIdentifyThread import HandIdentifyThread

# from PyQt6.QtGui import *
# from PyQt6.QtCore import *
# from PyQt6.QtWidgets import *

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建 UI 对象并设置界面
        self.ui = Ui_Basketball()
        self.ui.setupUi(self)

        self.timer1 = QTimer()
        self.timer2 = QTimer()
        self.timerflag1 = 3
        self.timerflag2 = 59
        self.page = 0

        #按键判断模式
        self.ui.AnalyseMode.clicked.connect(self.analyse_mode)
        self.ui.TrainMode.clicked.connect(self.train_mode)
        self.ui.BasketballMode.clicked.connect(self.basketball_mode)
        self.ui.HandMode.clicked.connect(self.hand_mode)
        self.ui.Back1.clicked.connect(self.video_select)
        self.ui.Back2.clicked.connect(self.back)
        self.ui.Back5.clicked.connect(self.back)
        self.ui.Back6.clicked.connect(self.back)

        #计时器
        self.timer1.timeout.connect(self.count_down1)
        self.timer2.timeout.connect(self.count_down2)

        #手势判断模式
        self.thread1 = HandIdentifyThread()
        self.thread1.pose.connect(self.analyse_mode2)
        self.thread1.start()
    
    #按键判断
    def analyse_mode(self):
        self.changePage(1)

    def train_mode(self):
        self.changePage(3)

    def back(self):
        self.changePage(0)

    #手势判断
    def analyse_mode2(self,pose):
        if(self.page==0):
            #跳转视频分析界面
            if(pose==1):
                self.changePage(1)
            #跳转训练界面
            if(pose==2):
                self.changePage(3)

        elif(self.page==2):
            #跳转回主界面
            if(pose==1):
                self.changePage(0)
            #停留在结算界面
            if(pose==2):
                self.changePage(2)

        elif(self.page==3):
            #进入控球训练
            if(pose==1):
                self.basketball_mode()
            #进入触点训练
            if(pose==2):
                self.hand_mode()

        elif(self.page == 4 or self.page == 5):
            #返回主页面
            if(pose==1):
                self.changePage(0)
            #返回训练模式选择界面
            if(pose==2):
                self.changePage(3)

    #跳转到指定页面
    def changePage(self, page):
        self.ui.stackedWidget.setCurrentIndex(page)
        self.page = page
        #开启手势识别线程
        if page == 0 or page == 2 or page == 3:
            if self.thread1.isRunning() == True:
                self.thread1.stop()
                self.thread1.wait()
                self.thread1.start()
            else:
                self.thread1.start()
        #结束手势识别线程
        if page == 1 or page == 4 or page == 5:
            if self.thread1.isRunning() == True:
                self.thread1.stop()
                self.thread1.wait()

    #选择视频所在位置并进行分析
    def video_select(self):
        # self.ui.player.setSource(QFileDialog.getOpenFileUrl()[0])
        self.url=QFileDialog.getOpenFileUrl()[0].toString()
        self.shootThread = ShootThread(self.url)
        self.shootThread.start()
        self.shootThread.finished.connect(self.video_play)
        self.shootThread.update_frame.connect(self.update_frame)
    
    #视频结束时进入结算页面
    def video_play(self, hit,attempts,rate,speed,angle):
        self.ui.Hit2.setText(str(hit))
        self.ui.Total2.setText(str(attempts))
        self.ui.Rate2.setText("%.1f"%(rate*100)+"%")
        self.ui.Speed2.setText("%.2fs"%speed)
        self.ui.Angle2.setText("%.1f°"%angle)
        self.changePage(2)
        
    #更新分析界面
    def update_frame(self, frame, hit, attempts):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.ui.AnalyseVideo.setPixmap(pixmap.scaled(self.ui.AnalyseVideo.size(), Qt.KeepAspectRatio))
        self.ui.Hit.setText(str(hit)+'  /  '+str(attempts))

    #控球训练模式
    def basketball_mode(self):
        self.changePage(4)
        #开启控球训练线程
        self.timerflag1 = 3
        self.timerflag2 = 59
        self.timer2.stop()
        self.timer1.start(1000)
        self.ui.Time1.setText("01:00")

    #触点训练模式
    def hand_mode(self):
        self.changePage(5)
        #开启控球训练线程
        self.timerflag1 = 3
        self.timerflag2 = 59
        self.timer1.stop()
        self.timer2.start(1000)
        self.ui.Time2.setText("01:00")

    #训练画面更新
    def train_frame(self, frame, points):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.ui.AnalyseVideo.setPixmap(pixmap.scaled(self.ui.AnalyseVideo.size(), Qt.KeepAspectRatio))
        self.ui.Points1.setText(str(points))

    #控球训练计时器
    def count_down1(self):
        if self.timerflag1>0:
            self.ui.BasketballIcon1.setText(str(self.timerflag1))
            self.timerflag1 -= 1
        else:
            self.ui.BasketballIcon1.setText("")
            #控球训练开始信号
            self.ui.Time1.setText("00:%02d"%self.timerflag2)
            self.timerflag2 -= 1

        if(self.timerflag2 == -1):
            self.timer1.stop()
            if self.thread1.isRunning() == False:
                 self.thread1.start()
            #控球训练结束信号

    #触点训练计时器
    def count_down2(self):
        if self.timerflag1>0:
            self.ui.HandIcon1.setText(str(self.timerflag1))
            self.timerflag1 -= 1
        else:
            self.ui.HandIcon1.setText("")
            #触点训练开始信号
            self.ui.Time2.setText("00:%02d"%self.timerflag2)
            self.timerflag2 -= 1

        if(self.timerflag2 == -1):
            self.timer2.stop()
            if self.thread1.isRunning() == False:
                 self.thread1.start()
            #触点训练结束信号

if __name__ == '__main__':
    # 运行界面
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())