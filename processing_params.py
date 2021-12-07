from matplotlib.pyplot import gray
from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

import cv2
import numpy as np

### my classes ###
from bubble_contour import *
from misc_methods import cvt_frame_color


class Process(Parameter):
    def __init__(self, **opts):
        opts["removable"] = True
        super().__init__(**opts)

    def process(self, frame, colorspace):
        return frame, colorspace

    def annotate(self, frame, colorspace):
        return frame, colorspace

    def __repr__(self):
        msg = self.opts["name"] + " Process"
        for c in self.childs:
            msg += f"\n{c.name()}: {c.value()}"
        return msg


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

    def process(self, frame, colorspace):
        self.bubbles = get_contours(gray=cvt_frame_color(frame, colorspace, "gray"),
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
        return frame, colorspace

    def annotate(self, frame, colorspace):
        try:
            return draw_annotations(
                frame=cvt_frame_color(frame, colorspace, "bgr"),
                bubbles=self.bubbles,
                min=self.lower_bound,
                max=self.upper_bound,
                highlight_idx=self.child("Overlay",
                                         "Bubble Highlight").value(),
                circum_color=self.child("Overlay",
                                        "Circumference Color").value(),
                center_color=self.child("Overlay", "Center Color").value(),
                neighbor_color=self.child("Overlay", "Neighbor Color").value(),
            ), "bgr"
        except AttributeError:
            return frame, colorspace


class AnalyzeBubblesWatershed(Process):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = "BubblesWatershed"

    def __init__(self, url, **opts):
        # if opts["type"] is not specified here,
        # type will be filled in during saveState()
        # opts["type"] = self.cls_type
        opts["url"] = url
        # only set these params not passed params already
        if "name" not in opts:
            opts["name"] = "BubbleWatershed"
        if "children" not in opts:
            opts["children"] = [{
                "name": "Toggle",
                "type": "bool",
                "value": True
            }, {
                "title":
                "View List",
                "name":
                "view_list",
                "type":
                "list",
                "value":
                "final",
                "limits": [
                    "gray",
                    "thresh",
                    "bg",
                    "fg",
                    "dist",
                    "unknown",
                    "final",
                ],
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
                "children": [{
                    "name": "Toggle",
                    "type": "bool",
                    "value": True
                }]
            }]
        super().__init__(**opts)

    def process(self, frame, colorspace):
        print("start", colorspace)
        gray = cvt_frame_color(frame, colorspace, "gray")
        _, self.thresh = cv2.threshold(gray,
                                       self.child("Lower").value(),
                                       self.child("Upper").value(),
                                       cv2.THRESH_BINARY_INV)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # expanded threshold to indicate outer bounds of interest
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(self.thresh,
                             kernel,
                             iterations=self.child("bg_iter").value())

        # Use distance transform then threshold to find points
        # within the bounds that could be used as seed
        # for watershed
        dt = cv2.distanceTransform(self.thresh, cv2.DIST_L2,
                                   self.child("dist_iter").value())
        # cv2.imshow("Dist Trans", dist_transform)
        ret, sure_fg = cv2.threshold(dt,
                                     self.child("fg_scale").value() * dt.max(),
                                     255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        # Labels connected components from 0 - n
        # 0 is for background
        count, markers = cv2.connectedComponents(sure_fg)
        print("cc ret:", count)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        print("Markers:", markers)

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv2.watershed(cvt_frame_color(frame, colorspace, "bgr"), markers)
        # border is -1
        # bg is 1
        # bubbles is >1
        # self.annotated = cv2.cvtColor(np.uint8(marker_show), cv2.COLOR_GRAY2BGR)

        print("IN annotate", colorspace)
        self.annotated = cvt_frame_color(frame, colorspace, "hsv")
        
        h, w, _ = frame.shape
        
        # draw hue identified contours
        for y in range(0, h):
            for x in range (0, w): 
                if markers [y, x] == -1:    # border
                    hue = 0
                    sat = 0
                    val = 255
                elif markers [y, x] == 1:   #bg
                    hue = 0
                    sat = 0
                    val = 0
                else:   # bubble
                    hue = (markers[y, x]) * (255 / count)
                    sat = 200
                    val = 200
                self.annotated[y, x] = [hue, sat, val]

        print("ehehee")
        self.annotated = cv2.cvtColor(self.annotated, "hsv", "bgr")
        print('ohohohoh')
        return frame, colorspace

    def annotate(self, frame, colorspace):
        
        return self.annotated, colorspace
