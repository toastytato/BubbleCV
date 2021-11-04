from enum import auto
from pyqtgraph.parametertree.parameterTypes import SliderParameter, GroupParameter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.ParameterTree import ParameterTree
from PyQt5.QtCore import QObject, QSettings, pyqtSignal


### my classes ###
from my_cv_process import *

RESET_DEFAULT_PARAMS = True


class Filter(Parameter, QObject):
    finished = pyqtSignal()

    def __init__(self, **opts):
        opts["removable"] = True
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
            {"name": "Toggle", "type": "bool", "value": False},
            SliderParameter(name="Upper", value=0, limits=(0, 255)),
            SliderParameter(name="Lower", value=255, limits=(0, 255)),
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
            {"name": "Toggle", "type": "bool", "value": False},
            SliderParameter(name="Iterations", limits=(0, 255)),
        ]
        super().__init__(**opts)

    def process(self, frame):
        return dilate(frame=frame, iterations=self.child("Iterations").value())


class Erode(Filter):
    def __init__(self, **opts):
        opts["name"] = "Erode"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": False},
            SliderParameter(name="Iterations", limits=(0, 255)),
        ]
        super().__init__(**opts)

    def process(self, frame):
        return erode(frame, iterations=self.child("Iterations").value())


class Invert(Filter):
    def __init__(self, **opts):
        opts["name"] = "Invert"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": False},
        ]
        super().__init__(**opts)

    def process(self, frame):
        return invert(frame)


class FilterGroup(GroupParameter):
    def __init__(self, **opts):
        # opts["name"] = "Filters"
        self.filters = {
            "Threshold": Threshold(),
            "Dilate": Dilate(),
            "Erode": Erode(),
            "Invert": Invert(),
        }

        opts["type"] = "group"
        opts["addText"] = "Add"
        opts["addList"] = self.filters.keys()
        super().__init__(**opts)

    def addNew(self, typ):
        self.addChild(self.filters[typ], autoIncrementName=True)


class MyParams(ParameterTree):
    paramChange = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.name = "MyParams"
        self.settings = QSettings("Bubble Deposition", self.name)
        params = [
            FilterGroup(name="Filters", children=[Threshold()]),
            {
                "name": "Analyze Circles",
                "type": "group",
                "children": [
                    {"name": "Toggle", "type": "bool", "value": True},
                    {"name": "Min Size", "type": "float", "value": 50},
                    {"name": "Bounds Offset", "type": "float", "value": 15},
                    {"name": "Num Neighbors", "type": "int", "value": 4},
                ],
            },
            {
                "name": "Overlay",
                "type": "group",
                "children": [
                    {"name": "Toggle", "type": "bool", "value": True},
                    {"name": "Bubble Highlight", "type": "int", "value": 0},
                    {"name": "Center Color", "type": "color", "value": "#ff0000"},
                    {
                        "name": "Circumference Color",
                        "type": "color",
                        "value": "#2CE2EE",
                    },
                    {"name": "Neighbor Color", "type": "color", "value": "#2C22EE"},
                    SliderParameter(
                        name="Mask Weight",
                        value=0,
                        step=0.01,
                        limits=(0, 1),
                    ),
                    {"name": "Export Distances", "type": "action"},
                ],
            },
        ]

        self.params = Parameter.create(name=self.name, type="group", children=params)

        # load saved data when available or otherwise specified in config.py
        # if self.settings.value("State") != None and not RESET_DEFAULT_PARAMS:
        #     self.state = self.settings.value("State")
        #     self.params.restoreState(self.state)
        # else:
        #     print("Loading default params for", self.name)

        self.setParameters(self.params, showTop=False)
        self.params.sigTreeStateChanged.connect(self.send_change)

    def send_change(self, param, changes):
        self.paramChange.emit(param, changes)

    # Convienience methods for modifying parameter values.
    def get_param_value(self, *childs):
        """Get the current value of a parameter."""
        return self.params.param(*childs).value()

    def set_param_value(self, value, *childs):
        """Set the current value of a parameter."""
        return self.params.param(*childs).setValue(value)

    def save_settings(self):
        self.state = self.params.saveState()
        self.settings.setValue("State", self.state)

    def print(self):
        print(self.name)
        print(self.params)
