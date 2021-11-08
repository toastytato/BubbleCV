import cv2


### Filtering ###

## Notes: Stop using kwargs for everything, leads to confusion down the line
def dilate(frame, iterations):
    return cv2.dilate(frame, None, iterations=iterations)


def erode(frame, iterations):
    return cv2.erode(frame, None, iterations=iterations)


def gaussian_blur(frame, radius, iterations):
    radius = radius * 2 + 1  # have only odd radius, needed for the blur kernel
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


if __name__ == "__main__":
    pass
