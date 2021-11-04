from PyQt5.QtCore import center
from PyQt5.QtWidgets import QWidget
import cv2
from PIL import ImageColor
import numpy as np
import imutils
from dataclasses import dataclass, field
import scipy.spatial as spatial
import pandas as pd
import math

### Filtering ###


def dilate(frame, **kwargs):
    frame = cv2.dilate(frame, None, iterations=kwargs["iterations"])
    return frame


def erode(frame, **kwargs):
    frame = cv2.erode(frame, None, iterations=kwargs["iterations"])
    return frame


def gaussian_blur(frame, **kwargs):
    for i in kwargs["iterations"]:
        frame = cv2.GaussianBlur(
            frame, (kwargs["radius"], kwargs["radius"]), cv2.BORDER_DEFAULT
        )
    return frame


def threshold(frame, **kwargs):
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


def invert(frame, **kwargs):
    return cv2.bitwise_not(frame)


### Processing ###


def get_contours(frame, **kwargs):
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubbles = []
    # only proceed if at least one contour was found
    if len(cnts) > 0:
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


def get_bounds(bubbles, **kwargs):
    centers = [(b.x, b.y) for b in bubbles]
    print(centers)
    lower_bound = np.min(centers, axis=0) + kwargs["offset"]
    upper_bound = np.max(centers, axis=0) - kwargs["offset"]
    in_id = 0
    out_id = -1
    for b in bubbles:
        if (
            b.x > lower_bound[0]
            and b.x < upper_bound[0]
            and b.y > lower_bound[1]
            and b.y < upper_bound[1]
        ):
            b.id = in_id
            in_id += 1
        else:
            b.id = out_id
            out_id -= 1

    return lower_bound, upper_bound


def get_neighbors(bubbles, **kwargs):
    centers = [(b.x, b.y) for b in bubbles]
    kd_tree = spatial.KDTree(data=centers)
    # num_neighbors + 1 to include the reference bubble
    dist_list, neighbor_idx_list = kd_tree.query(centers, k=kwargs["num_neighbors"] + 1)

    for neighbor_set, dist_set in zip(neighbor_idx_list, dist_list):

        center_idx = neighbor_set[0]  # = i
        x = bubbles[center_idx].x
        y = bubbles[center_idx].y

        if bubbles[center_idx].id >= 0:
            neighbors = []
            distances = []
            angles = []
            for n, d in zip(neighbor_set[1:], dist_set[1:]):
                neighbors.append(bubbles[n])
                distances.append(d)
                angle = math.degrees(math.atan2(bubbles[n].y - y, bubbles[n].x - x))
                angles.append(angle)

            bubbles[center_idx].neighbors = neighbors
            bubbles[center_idx].distances = distances
            bubbles[center_idx].angles = angles


def draw_annotations(frame, bubbles, min, max, **kwargs):
    # draw bounds
    cv2.rectangle(
        frame, (int(min[0]), int(min[1])), (int(max[0]), int(max[1])), (100, 24, 24), 3
    )

    sel_bubble = None
    for b in bubbles:
        if b.id >= 0:
            if b.id == kwargs["highlight_idx"]:
                sel_bubble = b
                continue
            # highlight all bubbles within bounds
            rgb = ImageColor.getcolor(kwargs["circum_color"], "RGB")
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.circle(frame, (int(b.x), int(b.y)), int(b.diameter / 2), bgr, 3)

    # highlight selected and neighbors
    if sel_bubble is not None:
        rgb = ImageColor.getcolor(kwargs["center_color"], "RGB")
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.circle(
            frame,
            (int(sel_bubble.x), int(sel_bubble.y)),
            int(sel_bubble.diameter / 2),
            bgr,
            3,
        )
        print(np.around(sel_bubble.angles, 2))

        for n in sel_bubble.neighbors:
            rgb = ImageColor.getcolor(kwargs["neighbor_color"], "RGB")
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.circle(frame, (int(n.x), int(n.y)), int(n.diameter / 2), bgr, 3)

    return frame


def export_csv(bubbles, url):

    df = pd.DataFrame()
    df["id"] = [b.id for b in bubbles if b.id >= 0]
    df["x"] = [b.x for b in bubbles if b.id >= 0]
    df["y"] = [b.y for b in bubbles if b.id >= 0]
    df["diameter"] = [b.diameter for b in bubbles if b.id >= 0]
    df["neighbors"] = [[n.id for n in b.neighbors] for b in bubbles if b.id >= 0]
    df["distances"] = [np.around(b.distances, 2) for b in bubbles if b.id >= 0]
    df["angles"] = [np.around(b.angles, 2) for b in bubbles if b.id >= 0]

    print(df)

    df.to_csv(f"{url}_processed.csv", index=False)


@dataclass
class Bubble:
    x: float
    y: float
    diameter: float
    id: int = -1
    neighbors: list = None
    distances: list = None
    angles: list = None


if __name__ == "__main__":
    pass
