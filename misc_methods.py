import cv2
import numpy as np
from pyqtgraph.parametertree import registerParameterType
'''
Methods shared between multiple files
'''

# frame = (cv2.imread(), 'gray')

# register my custom param types on instantiation of the class
def register_my_param(param_obj):
    print("Registering:", param_obj.cls_type)
    registerParameterType(param_obj.cls_type, param_obj)
    return param_obj


class MyFrame:
    clr_map = {
        "gray": {
            "bgr": cv2.COLOR_GRAY2BGR,
        },
        "bgr": {
            "gray": cv2.COLOR_BGR2GRAY,
            "hsv": cv2.COLOR_BGR2HSV
        },
        "hsv": {
            "bgr": cv2.COLOR_HSV2BGR
        }
    }

    def __init__(self, frame, color):
        self.f = frame
        self.clr = color

    def as_clr(color):

        return


def cvt_frame_color(frame, start, end):
    if start == end:
        return frame

    color_map = {
        "bgr": {
            "gray": cv2.COLOR_BGR2GRAY,
            "hsv": cv2.COLOR_BGR2HSV,
            "rgb": cv2.COLOR_BGR2RGB
        },
        "gray": {
            "bgr": cv2.COLOR_GRAY2BGR,
            "rgb": cv2.COLOR_GRAY2RGB,
        }
    }
    if start == "gray":
        frame = np.uint8(frame)
    return np.uint8(cv2.cvtColor(frame, color_map[start][end]))
