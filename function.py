import cv2
from ultralytics import YOLO
import numpy as np


def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)


def getAngleFromDatum(datum):
    size = len(datum.keypoints.xy[0])
    if size!= 0:
        hipX, hipY = datum.keypoints.xy[0][12].cpu()
        kneeX, kneeY = datum.keypoints.xy[0][13].cpu()
        ankleX, ankleY = datum.keypoints.xy[0][15].cpu()

        shoulderX, shoulderY = datum.keypoints.xy[0][5].cpu()
        elbowX, elbowY = datum.keypoints.xy[0][7].cpu()
        wristX, wristY = datum.keypoints.xy[0][9].cpu()

        # test = np.array([hipX, hipY])

        kneeAngle = calculateAngle(np.array([hipX, hipY]), np.array(
            [kneeX, kneeY]), np.array([ankleX, ankleY]))
        elbowAngle = calculateAngle(np.array([shoulderX, shoulderY]), np.array(
            [elbowX, elbowY]), np.array([wristX, wristY]))

        elbowCoord = np.array([int(elbowX), int(elbowY)])
        kneeCoord = np.array([int(kneeX), int(kneeY)])
        return elbowAngle, kneeAngle, elbowCoord, kneeCoord
    else:
        return 0,0,0,0

def getSpeedFromDatum(datum, status, time, timescale):
    size = len(datum.keypoints.xy[0])
    if size != 0:
        earX, earY = datum.keypoints.xy[0][4]
        elbowX, elbowY = datum.keypoints.xy[0][8]
        wristX, wristY = datum.keypoints.xy[0][10]
        if status == 1 and wristY < elbowY:
            time = 0
            status = 0
        if wristY < earY-10 and status == 0:
            status = 1
        elif wristY < elbowY and status == 0 :
            time = time + timescale
        return status, time
    else:
        return 0,0
    

def getAverage(x):
    num = 0
    total = 0
    for i in range(len(x)):
        num += 1
        total += x[i]
    average = total/num
    return  average

