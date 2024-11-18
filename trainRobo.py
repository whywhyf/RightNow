

from roboflow import Roboflow
rf = Roboflow(api_key="JWxOLLw6YEwiNTZ9CuQU")
project = rf.workspace("mytem").project("baskeball")
dataset = project.version(1).download("yolov8")
