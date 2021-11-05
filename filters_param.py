from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

### my classes ###
from my_cv_process import *


# Notes:
# - Parent classes cannot hold attributes
class Filter(Parameter, QObject):
    swap_filter = pyqtSignal(str, str)

    def __init__(self, **opts):
        opts["removable"] = True
        opts["context"] = ["Move Up", "Move Down"]
        super().__init__(**opts)

    def contextMenu(self, direction):
        self.swap_filter.emit(self.name(), direction)

    def __repr__(self):
        msg = self.opts["name"] + " Filter"
        for c in self.childs:
            msg += f"\n{c.name()}: {c.value()}"
        return msg


class Threshold(Filter):
    def __init__(self, **opts):
        opts["name"] = "Threshold"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            SliderParameter(name="Upper", value=255, limits=(0, 255)),
            SliderParameter(name="Lower", value=0, limits=(0, 255)),
        ]
        super().__init__(**opts)

    def process(self, frame):
        return threshold(
            frame=frame,
            lower=self.child("Lower").value(),
            upper=self.child("Upper").value(),
        )


class Dilate(Filter):
    def __init__(self, **opts):
        opts["name"] = "Dilate"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
        ]
        super().__init__(**opts)

    def process(self, frame):
        return dilate(frame=frame, iterations=self.child("Iterations").value())


class Erode(Filter):
    def __init__(self, **opts):
        opts["name"] = "Erode"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
        ]
        super().__init__(**opts)

    def process(self, frame):
        return erode(frame, iterations=self.child("Iterations").value())


class Invert(Filter):
    def __init__(self, **opts):
        opts["name"] = "Invert"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
        ]
        super().__init__(**opts)

    def process(self, frame):
        return invert(frame)


class Edge(Filter):
    def __init__(self, **opts):
        opts["name"] = "Canny Edge"
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
    def __init__(self, **opts):
        opts["name"] = "Gaussian Blur"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            {"name": "Radius", "type": "int", "value": 1, "limits": (0, 100)},
            {"name": "Iterations", "type": "int", "value": 0, "limits": (0, 50)},
        ]
        super().__init__(**opts)

    def process(self, frame):
        # radius
        print("Gaussiaingin")
        return gaussian_blur(
            frame,
            radius=self.child("Radius").value(),
            iterations=self.child("Iterations").value(),
        )
