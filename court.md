yolo

- pose用于识别人体姿势

- detect识别篮球、篮筐和人的位置

- segment如果位置精度不够，可以用segment提高精度



meidapipe

姿势识别帧数很高



yolo可以自定义训练集



- 识别用yolov8n.pt
- 姿势用yolov8n-pose.pt



- 自定义数据集训练yolo

  https://www.qbitai.com/2023/01/41203.html

- 公开篮球检测模型

  https://universe.roboflow.com/aidatasets-qwszk/datasets-iztee



- 球筐只需稳定检测10s，定下位置，后续无需检测

  https://universe.roboflow.com/basketball-z8lzd/basketball-6phla/model/3

- 远景篮球检测

  https://universe.roboflow.com/mytem/baskeball/model/1

- 动态篮球检测较好的

  https://universe.roboflow.com/meva/ballsdetectiontest/model/2

  https://universe.roboflow.com/mytem/baskeball/model/1



自定义数据集训练

https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#preparing-a-custom-dataset-for-yolov8

https://www.youtube.com/watch?v=wuZtUMEiKWY

https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=Y8cDtxLIBHgQ