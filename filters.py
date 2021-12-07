import cv2
import numpy as np

### Filtering ###


## Notes: Stop using kwargs for everything, leads to confusion down the line
def dilate(frame, iterations):
    return cv2.dilate(frame, None, iterations=iterations)


def erode(frame, iterations):
    return cv2.erode(frame, None, iterations=iterations)


def blur(frame, radius, iterations, view):
    radius = radius * 2 + 1  # have only odd radius, needed for the blur kernel
    kernel = (radius, radius)
    for i in range(iterations):
        if view == "Gaussian":
            frame = cv2.GaussianBlur(frame, kernel, cv2.BORDER_DEFAULT)
        elif view == "Median":
            frame = cv2.medianBlur(frame, radius)
        elif view == "Blur":
            frame = cv2.blur(frame, kernel)
    return frame


def threshold(frame, lower, upper, type):
    if type == "thresh":
        thresh_type = cv2.THRESH_BINARY
    elif type == "inv thresh":
        thresh_type = cv2.THRESH_BINARY_INV
    elif type == "otsu":
        thresh_type = cv2.THRESH_OTSU
    else:
        thresh_type = cv2.THRESH_OTSU

    _, frame = cv2.threshold(frame, lower, upper, thresh_type)

    return frame


def my_hough(frame, dp, min_dist, view=None):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist)
    print("Circles:", circles)
    # ensure at least some circles were found
    if circles is not None:
        print("circle found")
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255),
                          -1)

    return frame


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

    # print("Thresh:", thresh.shape, thresh.dtype)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = thresh

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=bg_iterations)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, dist_iter)
    # cv2.imshow("Dist Trans", dist_transform)
    ret, sure_fg = cv2.threshold(dist_transform,
                                 fg_scale * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero (boundaries)
    markers[unknown == 255] = 0

    markers = cv2.watershed(frame, markers)
    # frame[markers == -1] = [255, 0, 0]
    # markers == 0 does not exist apparently
    # markers == 1 is boundary
    # markers > 1 are ids of isolated contours
    show_markers = markers
    show_markers[markers == -1] = 0  # boundaries

    if view == "thresh":
        print("thresh")
        ret_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif view == "bg":
        print("bg")
        ret_frame = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)
    elif view == "fg":
        print("fg")
        ret_frame = np.uint8(cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR))
    elif view == "dist":
        print("dist", dist_transform)
        dist_transform = dist_transform * 255 / np.amax(dist_transform)
        print("max", np.amax(dist_transform), "min", np.amin(dist_transform))
        # make sure image is in uint8 to display gray scale properly (int, not flaot)
        ret_frame = np.uint8(cv2.cvtColor(dist_transform, cv2.COLOR_GRAY2BGR))
    elif view == "unknown":
        print("unknown")
        ret_frame = cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)
    elif view == "gray":
        print("gray")
        ret_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif view == "contours":
        print("contours")
        ret_frame = cv2.cvtColor(np.uint8(show_markers), cv2.COLOR_GRAY2BGR)
    else:  # view==None or view=="final":
        ret_frame = frame
        print("frame")

    print(ret_frame.shape, ret_frame.dtype)
    return ret_frame


def canny_edge(frame, thresh1, thresh2):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, thresh1, thresh2, (3, 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame


def invert(frame):
    return cv2.bitwise_not(frame)


if __name__ == "__main__":
    pass
