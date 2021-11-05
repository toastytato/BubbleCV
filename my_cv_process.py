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

## Notes: Stop using kwargs for everything, leads to confusion down the line
def dilate(frame, iterations):
    return cv2.dilate(frame, None, iterations=iterations)


def erode(frame, iterations):
    return cv2.erode(frame, None, iterations=iterations)


def gaussian_blur(frame, radius, iterations):
    radius = radius * 2 + 1
    for i in range(iterations):
        frame = cv2.GaussianBlur(frame, (radius, radius), cv2.BORDER_DEFAULT)
    return frame


def threshold(frame, lower, upper):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(frame, lower, upper, cv2.THRESH_BINARY)
    frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return frame


def canny_edge(frame, thresh1, thresh2):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, thresh1, thresh2, (3, 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame


def invert(frame):
    return cv2.bitwise_not(frame)


### Processing ###


def get_contours(frame, min):
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubbles = []
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        for i, c in enumerate(cnts):
            if cv2.contourArea(c) > min:
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


def get_bounds(bubbles, offset):
    centers = [(b.x, b.y) for b in bubbles]
    lower_bound = np.min(centers, axis=0) + offset
    upper_bound = np.max(centers, axis=0) - offset
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


def get_neighbors(bubbles, num_neighbors):
    centers = [(b.x, b.y) for b in bubbles]
    kd_tree = spatial.KDTree(data=centers)
    # num_neighbors + 1 to include the reference bubble
    dist_list, neighbor_idx_list = kd_tree.query(centers, k=num_neighbors + 1)

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


def draw_annotations(
    frame, bubbles, min, max, highlight_idx, circum_color, center_color, neighbor_color
):
    # draw bounds
    cv2.rectangle(
        frame, (int(min[0]), int(min[1])), (int(max[0]), int(max[1])), (100, 24, 24), 3
    )

    sel_bubble = None
    for b in bubbles:
        if b.id >= 0:
            if b.id == highlight_idx:
                sel_bubble = b
                continue
            # highlight all bubbles within bounds
            rgba = circum_color.getRgb()
            bgr = (rgba[2], rgba[1], rgba[0])
            cv2.circle(frame, (int(b.x), int(b.y)), int(b.diameter / 2), bgr, 3)

    # highlight selected and neighbors
    if sel_bubble is not None:
        rgba = center_color.getRgb()
        bgr = (rgba[2], rgba[1], rgba[0])
        cv2.circle(
            frame,
            (int(sel_bubble.x), int(sel_bubble.y)),
            int(sel_bubble.diameter / 2),
            bgr,
            3,
        )

        for n in sel_bubble.neighbors:
            rgba = neighbor_color.getRgb()
            bgr = (rgba[2], rgba[1], rgba[0])
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
