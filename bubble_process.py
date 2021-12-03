import math
import os
from typing import OrderedDict
import pandas as pd
import imutils
import cv2
import scipy.spatial as spatial
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

### Processing ###
# - Functions for processing the bubbles


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

                bubbles.append(Bubble(
                    x=x,
                    y=y,
                    diameter=radius * 2,
                ))
    # x, y, diameter, top, bottom, left, right

    return bubbles


def get_bounds(bubbles, scale_x, scale_y, offset_x, offset_y):
    centers = [(b.x, b.y) for b in bubbles]
    lower_bound = np.min(centers,
                         axis=0) - (scale_x - offset_x, scale_y - offset_y)
    upper_bound = np.max(centers,
                         axis=0) + (scale_x + offset_x, scale_y + offset_y)
    in_id = 0
    out_id = -1
    for b in bubbles:
        if (
                # b.x - b.diameter / 2 > lower_bound[0]
                # and b.x + b.diameter / 2 < upper_bound[0]
                # and b.y - b.diameter / 2 > lower_bound[1]
                # and b.y + b.diameter / 2 < upper_bound[1]
                b.x >= lower_bound[0] and b.x <= upper_bound[0]
                and b.y >= lower_bound[1] and b.y <= upper_bound[1]):
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
                angle = math.degrees(
                    math.atan2(bubbles[n].y - y, bubbles[n].x - x))
                angles.append(angle)

            bubbles[center_idx].neighbors = neighbors
            bubbles[center_idx].distances = distances
            bubbles[center_idx].angles = angles


def draw_annotations(frame, bubbles, min, max, highlight_idx, circum_color,
                     center_color, neighbor_color):
    # draw bounds
    cv2.rectangle(frame, (int(min[0]), int(min[1])),
                  (int(max[0]), int(max[1])), (100, 24, 24), 2)

    sel_bubble = None
    for b in bubbles:
        if b.id >= 0:
            if b.id == highlight_idx:
                sel_bubble = b
                continue
            # highlight all bubbles within bounds
            rgba = circum_color.getRgb()
            bgr = (rgba[2], rgba[1], rgba[0])
            cv2.circle(frame, (int(b.x), int(b.y)), int(b.diameter / 2), bgr,
                       2)

    # highlight selected and neighbors
    if sel_bubble is not None:
        rgba = center_color.getRgb()
        bgr = (rgba[2], rgba[1], rgba[0])
        cv2.circle(
            frame,
            (int(sel_bubble.x), int(sel_bubble.y)),
            int(sel_bubble.diameter / 2),
            bgr,
            2,
        )

        for n in sel_bubble.neighbors:
            rgba = neighbor_color.getRgb()
            bgr = (rgba[2], rgba[1], rgba[0])
            cv2.circle(frame, (int(n.x), int(n.y)), int(n.diameter / 2), bgr,
                       2)

    return frame


def export_csv(bubbles, conversion, url):

    df = pd.DataFrame()
    df["id"] = [b.id for b in bubbles if b.id >= 0]
    df["x"] = [b.x for b in bubbles if b.id >= 0]
    df["y"] = [b.y for b in bubbles if b.id >= 0]
    df["diameter"] = [b.diameter for b in bubbles if b.id >= 0]
    df["neighbors"] = [[n.id for n in b.neighbors] for b in bubbles
                       if b.id >= 0]
    df["distances"] = [np.around(b.distances, 2) for b in bubbles if b.id >= 0]
    df["angles"] = [np.around(b.angles, 2) for b in bubbles if b.id >= 0]
    df["units"] = "pixels"
    df["um/px"] = conversion

    name = os.path.splitext(url)[0]
    name = os.path.basename(name)

    path = get_save_dir('analysis', url) + "/datapoints.csv"
    df.to_csv(path, index=False)


def export_boxplots(bubbles, num_neighbors, conversion, url):

    #
    fig, ax = plt.subplots()

    diam_vs_dist = {}

    for b in bubbles:
        if b.id >= 0:
            diam = np.rint(b.diameter * conversion)
            if diam not in diam_vs_dist:
                diam_vs_dist[diam] = [d * conversion for d in b.distances]
            else:
                diam_vs_dist[diam].extend(
                    [d * conversion for d in b.distances])

    sorted_diam_vs_dist = sorted(diam_vs_dist.items())

    ax.boxplot([b[1] for b in sorted_diam_vs_dist])
    ax.set_xticklabels([b[0] for b in sorted_diam_vs_dist])

    ax.set_ylabel(f"Distances to {num_neighbors} nearest neighbors (um)")
    ax.set_xlabel("Nearest Integer Diameter (um)")

    # plt.show()
    plt.tight_layout()
    # plt.ioff()

    path = get_save_dir('analysis', url) + "/diam_vs_dist_box.png"
    plt.savefig(path)
    plt.show()


def export_scatter(bubbles, num_neighbors, conversion, url):
    diam = [b.diameter * conversion for b in bubbles if b.id >= 0]
    distances = [np.around(b.distances, 2) for b in bubbles if b.id >= 0]
    distances = [d * conversion for d in distances]

    fig, ax = plt.subplots()

    for d, l in zip(diam, distances):
        ax.scatter([d] * len(l), l)

    ax.set_ylabel(f"Distances to {num_neighbors} nearest neighbors (um)")
    ax.set_xlabel("Diameter (um)")

    plt.tight_layout()
    # plt.ioff()

    path = get_save_dir('analysis', url) + "/dist_vs_diam_scatter.png"

    plt.savefig(path)
    plt.show()


def export_dist_histogram(bubbles, num_neighbors, conversion, url):
    fig, ax = plt.subplots()

    dist = []
    for b in bubbles:
        if b.id >= 0:
            dist = np.append(dist, [d * conversion for d in b.distances])

    bins = np.arange(0, np.amax(dist), 1)
    ax.hist(dist, bins)

    ax.set_ylabel("Number of distances")
    ax.set_xlabel("Nearest Integer Distances (um)")

    path = get_save_dir('analysis', url) + "/dist_histogram.png"
    plt.savefig(path)
    plt.show()


def export_diam_histogram(bubbles, num_neighbors, conversion, url):
    fig, ax = plt.subplots()

    diam = []
    for b in bubbles:
        if b.id >= 0:
            diam = np.append(diam, b.diameter)

    print()
    bins = np.arange(0, np.amax(diam), 1)
    ax.hist(diam, bins)

    ax.set_ylabel("Number of diameters")
    ax.set_xlabel("Nearest Integer Diameter (um)")

    # if not os.path.exists('analysis/histograms'):
    #     os.makedirs('analysis/histograms')
    # name = os.path.splitext(url)[0]
    # name = os.path.basename(name)
    path = get_save_dir('analysis', url) + "/diam_histogram.png"
    plt.savefig(path)
    plt.show()


def get_save_dir(main_path, url):
    name = os.path.splitext(url)[0]
    name = os.path.basename(name)
    path = f'{main_path}/{name}'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@dataclass
class Bubble:
    x: float
    y: float
    diameter: float
    id: int = -1
    neighbors: list = None
    distances: list = None
    angles: list = None


# ------- Watershed Functions ---------------

def my_watershed(frame,
                 lower,
                 upper,
                 fg_scale,
                 bg_iterations,
                 dist_iter,
                 view=None):
    print(view)

    # frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, lower, upper, cv2.THRESH_BINARY_INV)

    print("Thresh:", thresh.shape, thresh.dtype)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = thresh

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=bg_iterations)

    # Finding sure foreground area
    dt = cv2.distanceTransform(opening, cv2.DIST_L2, dist_iter)
    # cv2.imshow("Dist Trans", dist_transform)
    ret, sure_fg = cv2.threshold(dt, fg_scale * dt.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    print("Markers:", markers)

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [255, 0, 0]

    if view == "thresh":
        print("thresh")
        ret_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif view == "morph":
        print("morph")
        ret_frame = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    elif view == "bg":
        print("bg")
        ret_frame = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)
    elif view == "fg":
        print("fg")
        ret_frame = np.uint8(cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR))
    elif view == "dist":
        print("dist", dt)
        # dist_transform = dist_transform * 255 / np.amax(dist_transform)
        print("max", np.amax(dt), "min", np.amin(dt))
        # make sure image is in uint8 to display gray scale properly (int, not flaot)
        ret_frame = np.uint8(cv2.cvtColor(dt, cv2.COLOR_GRAY2BGR))
    elif view == "unknown":
        print("unknown")
        ret_frame = cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
    elif view == "gray":
        print("gray")
        ret_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        print(ret_frame)
    else:  # view==None or view=="final":
        ret_frame = frame
        print("frame")

    print(ret_frame.shape, ret_frame.dtype)
    return ret_frame