import cv2
import numpy as np
'''
Methods shared between multiple files
'''


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
