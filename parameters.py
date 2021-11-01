from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.ParameterTree import ParameterTree
from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import QSettings

RESET_DEFAULT_PARAMS = True


class MyParams(ParameterTree):
    paramChange = QtCore.pyqtSignal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.name = "MyParams"
        self.settings = QSettings("Bubble Deposition", self.name)
        params = [
            # {
            #     "name": "Gaussian Blur",
            #     "type": "group",
            #     "children": [
            #         {"name": "Toggle", "type": "bool", "value": False},
            #         {"name": "Radius", "type": "float", "value": 0, "limits": (0, 10)},
            #         {
            #             "name": "Iterations",
            #             "type": "int",
            #             "value": 0,
            #             "limits": (0, 10),
            #         },
            #     ],
            # },
            {
                "name": "Invert",
                "type": "group",
                "children": [
                    {"name": "Toggle", "type": "bool", "value": True},
                ],
            },
            {
                "name": "Blob Filter",
                "type": "group",
                "children": [
                    {"name": "Toggle", "type": "bool", "value": False},
                    {"name": "Erode", "type": "int", "value": 0, "limits": (0, 10)},
                    {"name": "Dilate", "type": "int", "value": 0, "limits": (0, 10)},
                ],
            },
            # {
            #     "name": "Threshold",
            #     "type": "group",
            #     "children": [
            #         {"name": "Toggle", "type": "bool", "value": False},
            #         {"name": "Lower", "type": "int", "value": 0, "limits": (0, 255)},
            #         {"name": "Upper", "type": "int", "value": 0, "limits": (0, 255)},
            #     ],
            # },
            {
                "name": "Analyze Circles",
                "type": "group",
                "children": [
                    {"name": "Toggle", "type": "bool", "value": True},
                    {"name": "Min Size", "type": "float", "value": 50},
                    {"name": "Bounds Offset", "type": "float", "value": 15},
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
                    {"name": "Highlight Color", "type": "color", "value": "#2C22EE"},
                    {
                        "name": "Mask Weight",
                        "type": "float",
                        "value": 0,
                        "step": 0.05,
                        "limits": (0, 1),
                    },
                    {"name": "Export Distances", "type": "action"},
                ],
            },
        ]

        self.params = Parameter.create(name=self.name, type="group", children=params)

        # load saved data when available or otherwise specified in config.py
        if self.settings.value("State") != None and not RESET_DEFAULT_PARAMS:
            self.state = self.settings.value("State")
            self.params.restoreState(self.state)
        else:
            print("Loading default params for", self.name)

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
