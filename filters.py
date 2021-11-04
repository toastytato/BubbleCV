from pyqtgraph.parametertree.parameterTypes import SliderParameter, GroupParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, QSettings, pyqtSignal

### my classes ###
from my_cv_process import *

class Filter(Parameter, QObject):
    finished = pyqtSignal()

    def __init__(self, **opts):
        opts["removable"] = True
        # opts["context"] = ["Move Up", "Move Down"]
        super().__init__(**opts)

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
            SliderParameter(name="Iterations", limits=(0, 10)),
        ]
        super().__init__(**opts)

    def process(self, frame):
        return dilate(frame=frame, iterations=self.child("Iterations").value())


class Erode(Filter):
    def __init__(self, **opts):
        opts["name"] = "Erode"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            SliderParameter(name="Iterations", limits=(0, 10), step=1),
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


class FilterGroup(GroupParameter):
    def __init__(self, **opts):
        # opts["name"] = "Filters"
        self.filters = {
            "Threshold": Threshold,
            "Dilate": Dilate,
            "Erode": Erode,
            "Invert": Invert,
        }

        opts["type"] = "group"
        opts["addText"] = "Add"
        opts["addList"] = self.filters.keys()
        super().__init__(**opts)

    def addNew(self, typ):
        filter = self.filters[typ]()
        self.addChild(filter, autoIncrementName=True)
