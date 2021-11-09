from os import name

from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

### my classes ###
from filters import *
from bubble_process import *

# Notes:
# - Parent classes cannot hold attributes
# BUG: When using pyqtgraph save and restores,
# it only restores parameters states,
# not any custom methods you made
# filter object becomes a regular parameter object with the filter params...
# how 2 fix? Idk
# eyy fixed make custom Parameter objects 11/8/21


class Filter(Parameter, QObject):
    swap_filter = pyqtSignal(str, str)

    def __init__(self, **opts):
        opts["removable"] = True
        opts["context"] = ["Move Up", "Move Down"]
        super().__init__(**opts)

    def contextMenu(self, direction):
        self.swap_filter.emit(self.name(), direction)

    def process(self, frame):
        return frame

    def annotate(self, frame):
        return frame

    def __repr__(self):
        msg = self.opts["name"] + " Filter"
        for c in self.childs:
            msg += f"\n{c.name()}: {c.value()}"
        return msg


class Threshold(Filter):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = "ThresholdFilter"

    def __init__(self, **opts):
        # if opts["type"] is not specified here,
        # type will be filled in during saveState()
        # opts["type"] = self.cls_type

        # only set these params not passed params already
        if "name" not in opts:
            opts["name"] = "Threshold"

        if "children" not in opts:
            opts["children"] = [
                {"name": "Toggle", "type": "bool", "value": True},
                {"name": "Upper", "type": "slider", "value": 255, "limits": (0, 255)},
                {"name": "Lower", "type": "slider", "value": 0, "limits": (0, 255)},
            ]
        super().__init__(**opts)

    def process(self, frame):
        return threshold(
            frame=frame,
            lower=self.child("Lower").value(),
            upper=self.child("Upper").value(),
        )


class Dilate(Filter):
    cls_type = "DilateFilter"

    def __init__(self, **opts):
        # opts["type"] = self.cls_type

        if "name" not in opts:
            opts["name"] = "Dilate"

        if "children" not in opts:
            opts["children"] = [
                {"name": "Toggle", "type": "bool", "value": True},
                {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
            ]
        super().__init__(**opts)

    def process(self, frame):
        return dilate(frame=frame, iterations=self.child("Iterations").value())


class Erode(Filter):
    cls_type = "ErodeFilter"

    def __init__(self, **opts):
        # opts["type"] = self.cls_type

        if "name" not in opts:
            opts["name"] = "Erode"

        if "children" not in opts:
            opts["children"] = [
                {"name": "Toggle", "type": "bool", "value": True},
                {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
            ]
        super().__init__(**opts)

    def process(self, frame):
        return erode(frame, iterations=self.child("Iterations").value())


class Invert(Filter):
    cls_type = "InvertFilter"

    def __init__(self, **opts):
        # opts["type"] = self.cls_type

        if "name" not in opts:
            opts["name"] = "Invert"

        if "children" not in opts:
            opts["children"] = [
                {"name": "Toggle", "type": "bool", "value": True},
            ]
        super().__init__(**opts)

    def process(self, frame):
        return invert(frame)


class Edge(Filter):
    cls_type = "EdgeFilter"

    def __init__(self, **opts):
        # opts["type"] = self.cls_type

        if "name" not in opts:
            opts["name"] = "Canny Edge"

        if "children" not in opts:
            opts["children"] = [
                {"name": "Toggle", "type": "bool", "value": True},
                SliderParameter(name="Thresh1", value=0, limits=(0, 255)),
                SliderParameter(name="Thresh2", value=0, limits=(0, 255)),
            ]
        super().__init__(**opts)

    def process(self, frame):
        return canny_edge(
            frame,
            thresh1=self.child("Thresh1").value(),
            thresh2=self.child("Thresh2").value(),
        )


class GaussianBlur(Filter):
    cls_type = "GaussianBlurFilter"

    def __init__(self, **opts):
        # opts["type"] = self.cls_type

        if "name" not in opts:
            opts["name"] = "Gaussian Blur"

        if "children" not in opts:
            opts["children"] = [
                {"name": "Toggle", "type": "bool", "value": True},
                {"name": "Radius", "type": "int", "value": 1, "limits": (0, 100)},
                {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
            ]

        super().__init__(**opts)

    def process(self, frame):
        # radius
        return gaussian_blur(
            frame,
            radius=self.child("Radius").value(),
            iterations=self.child("Iterations").value(),
        )
