from PyQt5.QtCore import center
from PyQt5.QtWidgets import QWidget
import cv2
from PIL import ImageColor
import numpy as np
import imutils
from dataclasses import dataclass, field
import scipy.spatial as spatial
import pandas as pd
from typing import List


def dilate(frame, **kwargs):
    frame = cv2.dilate(frame, None, iterations=kwargs["dilate"])
    return frame

def erode(frame, **kwargs):
    frame = cv2.erode(frame, None, iterations=kwargs["erode"])
    return frame

def gaussian_blur(frame, **kwargs):
    for i in kwargs["iterations"]:
        frame = cv2.GaussianBlur(
            frame, (kwargs["radius"], kwargs["radius"]), cv2.BORDER_DEFAULT
        )
    return frame


def thresh(frame, **kwargs):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(
        frame, kwargs["lower"], kwargs["upper"], cv2.THRESH_BINARY
    )
    frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return frame


def edge(frame):
    edges = cv2.Canny(frame, threshold1=100, threshold2=200)
    out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # return out


def hough(frame, **kwargs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(["int"])

        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return frame


def invert(frame, **kwargs):
    return cv2.bitwise_not(frame)


def contours(frame, **kwargs):
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubbles = []
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        # c = max(cnts, key=cv2.contourArea)
        for i, c in enumerate(cnts):
            if cv2.contourArea(c) > kwargs["min"]:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # M = cv2.moments(c)

                bubbles.append(
                    Bubble(
                        x=x,
                        y=y,
                        diameter=radius * 2,
                    )
                )
    # x, y, diameter, top, bottom, left, right

    return bubbles


def get_neighbor_distances(bubbles, **kwargs):
    centers = [(b.x, b.y) for b in bubbles]
    kd_tree = spatial.KDTree(data=centers)
    dist, idx = kd_tree.query(centers, k=5)

    return dist, idx


def draw_annotations(frame, bubbles, neighbor_idx, distances, **kwargs):
    # draw circle around circumference
    centers = [(b.x, b.y) for b in bubbles]
    min = np.min(centers, axis=0) + kwargs["offset"]
    max = np.max(centers, axis=0) - kwargs["offset"]

    cv2.rectangle(
        frame, (int(min[0]), int(min[1])), (int(max[0]), int(max[1])), (100, 24, 24), 3
    )

    for i, neighbor_set in enumerate(neighbor_idx):

        ref_bubble_idx = neighbor_set[0]  # = i

        x = bubbles[ref_bubble_idx].x
        y = bubbles[ref_bubble_idx].y
        radius = bubbles[ref_bubble_idx].diameter / 2

        # ignore bubbles out of bounds
        if x > min[0] and x < max[0] and y > min[1] and y < max[1]:
            # highlight bubble selected
            if ref_bubble_idx == kwargs["highlight_idx"]:
                rgb = ImageColor.getcolor(kwargs["center_color"], "RGB")
                bgr = (rgb[2], rgb[1], rgb[0])
                cv2.circle(frame, (int(x), int(y)), int(radius), bgr, 3)
            # highlight all bubbles within bounds
            else:
                rgb = ImageColor.getcolor(kwargs["circum_color"], "RGB")

            bubbles[i].dist = distances[i][1:]

            # highlight neighbors to selected bubbles
            if ref_bubble_idx == kwargs["highlight_idx"]:
                for idx, n in enumerate(neighbor_set[1:], start=1):
                    print(n)
                    x = bubbles[n].x
                    y = bubbles[n].y
                    radius = bubbles[n].diameter / 2
                    print(f"({x},{y}), r:{radius}, d:{distances[i][idx]}")
                    rgb = ImageColor.getcolor(kwargs["highlight_color"], "RGB")
                    bgr = (rgb[2], rgb[1], rgb[0])
                    cv2.circle(frame, (int(x), int(y)), int(radius), bgr, 3)
    export_csv(bubbles, distances)
    return frame


def export_csv(bubbles, distances):
    data = {
        "x": [b.x for b in bubbles],
        "y": [b.y for b in bubbles],
        "diameter": [b.diameter for b in bubbles],
        "dist0": [distances[i][1] for i, b in enumerate(bubbles)],
        "dist1": [distances[i][2] for i, b in enumerate(bubbles)],
        "dist2": [distances[i][3] for i, b in enumerate(bubbles)],
        "dist3": [distances[i][4] for i, b in enumerate(bubbles)],
    }
    df = pd.DataFrame(data)

    df.to_csv(f"Point Distances.csv")


@dataclass
class Bubble:
    x: float
    y: float
    diameter: float


if __name__ == "__main__":
    pass
