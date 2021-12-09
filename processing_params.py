from matplotlib.pyplot import gray
from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

import cv2
import numpy as np

### my classes ###
from bubble_contour import *
from misc_methods import MyFrame, register_my_param


class Process(Parameter):
    def __init__(self, **opts):
        opts["removable"] = True
        super().__init__(**opts)

    def process(self, frame):
        return frame

    def annotate(self, frame):
        return frame

    def __repr__(self):
        msg = self.opts["name"] + " Process"
        for c in self.childs:
            msg += f"\n{c.name()}: {c.value()}"
        return msg


@register_my_param
class AnalyzeBubbles(Process):
    cls_type = "Bubbles"

    def __init__(self, url, **opts):
        opts["url"] = url
        if "name" not in opts:
            opts["name"] = "Bubbles"
        if "children" not in opts:
            opts["children"] = [
                {
                    "name": "Toggle",
                    "type": "bool",
                    "value": True
                },
                {
                    "name": "Min Size",
                    "type": "slider",
                    "value": 50,
                    "limits": (0, 200),
                },
                {
                    "name": "Num Neighbors",
                    "type": "int",
                    "value": 4
                },
                {
                    "name": "Bounds Offset X",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Bounds Offset Y",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Bounds Scale X",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Bounds Scale Y",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Conversion",
                    "type": "float",
                    "units": "um/px",
                    "value": 600 / 900,
                    "readonly": True,
                },
                {
                    "name": "Export Distances",
                    "type": "action"
                },
                {
                    "name": "Export Graph",
                    "type": "action"
                },
                {
                    "name":
                    "Overlay",
                    "type":
                    "group",
                    "children": [
                        {
                            "name": "Toggle",
                            "type": "bool",
                            "value": True
                        },
                        {
                            "name": "Bubble Highlight",
                            "type": "int",
                            "value": 0
                        },
                        {
                            "name": "Center Color",
                            "type": "color",
                            "value": "#ff0000",
                        },
                        {
                            "name": "Circumference Color",
                            "type": "color",
                            "value": "#2CE2EE",
                        },
                        {
                            "name": "Neighbor Color",
                            "type": "color",
                            "value": "#2C22EE",
                        },
                    ],
                },
            ]
        super().__init__(**opts)
        self.bubbles = []
        self.url = url
        self.um_per_pixel = self.child("Conversion").value()
        self.child("Export Distances").sigActivated.connect(self.export_csv)
        self.child("Export Graph").sigActivated.connect(self.export_graphs)

        # self.sigTreeStateChanged.connect(self.on_change)

    def export_csv(self, change):
        # print("Export", change)

        if self.bubbles is not None:
            if self.url is None:
                export_csv(  # from bubble_processes
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url="exported_data",
                )
                print("Default Export")
            else:
                export_csv(
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url=self.url + "_data",
                )

    def export_graphs(self, change):
        print(self.url)
        export_boxplots(
            self.bubbles,
            self.child("Num Neighbors").value(),
            self.um_per_pixel,
            self.url,
        )
        export_scatter(
            self.bubbles,
            self.child("Num Neighbors").value(),
            self.um_per_pixel,
            self.url,
        )
        export_dist_histogram(self.bubbles,
                              self.child("Num Neighbors").value(),
                              self.um_per_pixel, self.url)
        export_diam_histogram(self.bubbles,
                              self.child("Num Neighbors").value(),
                              self.um_per_pixel, self.url)

    def process(self, frame):
        self.bubbles = get_contours(frame=frame,
                                    min=self.child("Min Size").value())
        if len(self.bubbles) > self.child("Num Neighbors").value():
            self.lower_bound, self.upper_bound = get_bounds(
                bubbles=self.bubbles,
                scale_x=self.child("Bounds Scale X").value(),
                scale_y=self.child("Bounds Scale Y").value(),
                offset_x=self.child("Bounds Offset X").value(),
                offset_y=self.child("Bounds Offset Y").value(),
            )
            get_neighbors(bubbles=self.bubbles,
                          num_neighbors=self.child("Num Neighbors").value()
                          )  # modifies param to assign neighbors to bubbles
        return frame

    def annotate(self, frame):
        try:
            return draw_annotations(
                frame=frame,
                bubbles=self.bubbles,
                min=self.lower_bound,
                max=self.upper_bound,
                highlight_idx=self.child("Overlay",
                                         "Bubble Highlight").value(),
                circum_color=self.child("Overlay",
                                        "Circumference Color").value(),
                center_color=self.child("Overlay", "Center Color").value(),
                neighbor_color=self.child("Overlay", "Neighbor Color").value(),
            )
        except AttributeError:
            return frame


@register_my_param
class AnalyzeBubblesWatershed(Process):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = "BubblesWatershed"

    def __init__(self, url, **opts):
        # if opts["type"] is not specified here,
        # type will be filled in during saveState()
        # opts["type"] = self.cls_type
        opts["url"] = url

        self.img = {
            "gray": None,
            "thresh": None,
            "bg": None,
            "fg": None,
            "dist": None,
            "unknown": None,
            "final": None,
        }

        # only set these params not passed params already
        if "name" not in opts:
            opts["name"] = "BubbleWatershed"
        if "children" not in opts:
            opts["children"] = [{
                "name": "Toggle",
                "type": "bool",
                "value": True
            }, {
                "name": "Upper",
                "type": "slider",
                "value": 255,
                "limits": (0, 255)
            }, {
                "name": "Lower",
                "type": "slider",
                "value": 0,
                "limits": (0, 255)
            }, {
                "title": "FG scale",
                "name": "fg_scale",
                "type": "slider",
                "value": 0.01,
                "precision": 4,
                "step": 0.0005,
                "limits": (0, 1),
            }, {
                "title": "BG Iterations",
                "name": "bg_iter",
                "type": "int",
                "value": 3,
                "limits": (0, 255),
            }, {
                "title": "Dist Transform Iter.",
                "name": "dist_iter",
                "type": "list",
                "value": 5,
                "limits": [0, 3, 5],
            }, {
                "name":
                "Overlay",
                "type":
                "group",
                "children": [
                    {
                        "name": "Toggle",
                        "type": "bool",
                        "value": True
                    },
                    {
                        "title": "View List",
                        "name": "view_list",
                        "type": "list",
                        "value": list(self.img.keys())[-1],
                        "limits": list(self.img.keys()),
                    },
                ]
            }]
        super().__init__(**opts)

    def process(self, frame):
        print("start processing")
        self.img["gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, self.img["thresh"] = cv2.threshold(self.img["gray"].f,
                                              self.child("Lower").value(),
                                              self.child("Upper").value(),
                                              cv2.THRESH_BINARY_INV)
        # expanded threshold to indicate outer bounds of interest
        kernel = np.ones((3, 3), np.uint8)
        self.img["bg"] = cv2.dilate(self.img["thresh"],
                                    kernel,
                                    iterations=self.child("bg_iter").value())
        # Use distance transform then threshold to find points
        # within the bounds that could be used as seed
        # for watershed
        self.img["dist"] = cv2.distanceTransform(
            self.img["thresh"], cv2.DIST_L2,
            self.child("dist_iter").value())
        # cv2.imshow("Dist Trans", dist_transform)
        _, self.img["fg"] = cv2.threshold(
            self.img["dist"],
            self.child("fg_scale").value() * self.img["dist"].max(), 255, 0)

        # Finding unknown region
        self.img["fg"] = np.uint8(self.img["fg"])
        self.img["unknown"] = cv2.subtract(self.img["bg"], self.img["fg"])

        # Marker labeling
        # Labels connected components from 0 - n
        # 0 is for background
        count, markers = cv2.connectedComponents(self.img["fg"])
        print("cc ret:", count)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        print("Markers:", markers)

        # Now, mark the region of unknown with zero
        markers[self.img["unknown"] == 255] = 0

        markers = cv2.watershed(frame, markers)
        # border is -1
        # 0 does not exist
        # bg is 1
        # bubbles is >1
        # self.annotated = cv2.cvtColor(np.uint8(marker_show), cv2.COLOR_GRAY2BGR)
        print("Np unique:", np.unique(markers))

        self.img["final"] = frame.copy()

        for label in np.unique(markers):
            # if the label is zero, we are examining the 'background'
            # if label is -1, it is the border and we don't need to label it
            # so simply ignore it
            if label == 1 or label == -1:
                continue
            # otherwise, allocate memory
            # for the label region and draw
            # it on the mask
            print("Label", label)
            mask = np.zeros(self.img['gray'].shape, dtype="uint8")
            mask[markers == label] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(self.img["final"], (int(x), int(y)), int(r),
                       (0, 255, 0), 1)
            cv2.putText(self.img["final"], "{}".format(label),
                        (int(x) - 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)

        # self.annotated[markers == -1] = [0, 0, 255]
        # self.annotated[markers == 1] = [0, 0, 0]

        # self.annotated = cv2.cvtColor(self.annotated, cv2.COLOR_HSV2BGR)
        return frame

    def annotate(self, frame):
        view = self.child('Overlay', 'view_list').value()
        print("View:", view)
        print("Shape:", self.img[view].shape)
        if view == "thresh":
            print("thresh")
            return cv2.cvtColor(self.img[view], cv2.COLOR_GRAY2BGR)
        elif view == "bg":
            print("bg")
            return cv2.cvtColor(self.img[view], cv2.COLOR_GRAY2BGR)
        elif view == "fg":
            print("fg")
            return np.uint8(cv2.cvtColor(self.img[view], cv2.COLOR_GRAY2BGR))
        elif view == "dist":
            print("dist")
            self.img[view] = self.img[view] * 255 / np.amax(self.img[view])
            return np.uint8(cv2.cvtColor(self.img[view], cv2.COLOR_GRAY2BGR))
        elif view == "unknown":
            print("unknown")
            return cv2.cvtColor(self.img[view], cv2.COLOR_GRAY2BGR)
        elif view == "gray":
            print("gray")
            return cv2.cvtColor(self.img[view], cv2.COLOR_GRAY2BGR)
        elif view == "final":
            print("final")
            return self.img[view]
