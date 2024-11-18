import numpy as np
import cv2
# 读入图片
img = cv2.imread('./images/3.png')
img_size = (img.shape[1], img.shape[0])
print(img_size)
# 确定需要矫正的区域，左上，左下，右下，右上
# src = np.float32([[50,590],[350,840],[1500,600],[880,560]])
src = np.float32([[535,280],[533,360],[642,372],[642,297]])
# 确定需要矫正成的形状，和上面一一对应 
dst = np.float32([[900,0],[900,100],[1020,100],[1020,0]])
# cv2.circle(img, (910, 580), 3, (0,0,255), 2)
cv2.circle(img, (642, 297), 3, (0,0,255), 2)

# 获取矫正矩阵，也就步骤
M = cv2.getPerspectiveTransform(src, dst)
print(M)

# 进行矫正，把img
newxy = np.dot(M, np.array([910- 960, 580 - 540, 1]).T)
print(newxy)

img = cv2.warpPerspective(img, M, img_size)
# 定义原图坐标点
original_points = np.array([[[910, 580]]], dtype=np.float32)

# 使用变换矩阵进行坐标转换
new_points = cv2.perspectiveTransform(original_points, M)

# 获取新图坐标
new_x = new_points[0][0][0]
new_y = new_points[0][0][1]

# 打印新图坐标
print("New coordinates: ", (new_x, new_y))
cv2.circle(img, (int(new_x), int(new_y)), 3, (0,0,255), 2)
# 展示校正后的图形
cv2.imshow('output', img)
cv2.waitKey(0)
