from enum import auto
from matplotlib.pyplot import step
from pyqtgraph.parametertree.parameterTypes import SliderParameter, GroupParameter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.ParameterTree import ParameterTree
from PyQt5.QtCore import QObject, QSettings, pyqtSignal


### my classes ###
from my_cv_process import *
from filters import *

RESET_DEFAULT_PARAMS = True


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
