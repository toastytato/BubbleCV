from os import name

from pyqtgraph.parametertree.parameterTypes import (
    SliderParameter,
    registerParameterType,
)
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

### my classes ###
from my_cv_process import *


# Notes:
# - Parent classes cannot hold attributes
# BUG: When using pyqtgraph save and restores,
# it only restores parameters states,
# not any custom methods you made
# filter object becomes a regular parameter object with the filter params...
# how 2 fix? Idk


class Filter(Parameter, QObject):
    swap_filter = pyqtSignal(str, str)

    def __init__(self, **opts):
        opts["removable"] = True
        opts["context"] = ["Move Up", "Move Down"]
        # opts["methods"] = {"process": self.process, "annotate": self.annotate}
        super().__init__(**opts)

    def contextMenu(self, direction):
        print("context pressed", type(self))
        self.swap_filter.emit(self.name(), direction)

    def process(self, frame):
        return frame

    def annotate(frame):
        return frame

    def __repr__(self):
        msg = self.opts["name"] + " Filter"
        for c in self.childs:
            msg += f"\n{c.name()}: {c.value()}"
        return msg


class Threshold(Filter):
    cls_type = "Threshold"

    def __init__(self, **opts):
        opts["name"] = "Threshold"
        opts["type"] = self.cls_type
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
    cls_type = "Dilate"

    def __init__(self, **opts):
        opts["name"] = "Dilate"
        opts["type"] = self.cls_type
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
        ]
        super().__init__(**opts)

    def process(self, frame):
        return dilate(frame=frame, iterations=self.child("Iterations").value())


class Erode(Filter):
    cls_type = "Erode"

    def __init__(self, **opts):
        opts["name"] = "Erode"
        opts["type"] = self.cls_type
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
        ]
        super().__init__(**opts)

    def process(self, frame):
        return erode(frame, iterations=self.child("Iterations").value())


class Invert(Filter):
    cls_type = "Invert"

    def __init__(self, **opts):
        opts["name"] = "Invert"
        opts["type"] = self.cls_type
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
        ]
        super().__init__(**opts)

    def process(self, frame):
        return invert(frame)


class Edge(Filter):
    cls_type = "Edge"

    def __init__(self, **opts):
        opts["name"] = "Canny Edge"
        opts["type"] = self.cls_type
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
    cls_type = "GaussianBlur"

    def __init__(self, **opts):
        opts["name"] = "Gaussian Blur"
        opts["type"] = self.cls_type
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
