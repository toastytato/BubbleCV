from os import DirEntry
from pyqtgraph.parametertree.parameterTypes import *
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.ParameterTree import ParameterTree
from PyQt5.QtCore import QObject, QSettings, pyqtSignal
from pprint import *
import pickle

### my classes ###
from my_cv_process import *
from filters_param import *
from processes_param import *

RESET_DEFAULT_PARAMS = True


class FilterGroup(GroupParameter):
    def __init__(self, **opts):
        # opts["name"] = "Filters"
        self.operations = {
            "Threshold": Threshold,
            "Dilate": Dilate,
            "Erode": Erode,
            "Invert": Invert,
            "Gaussian Blur": GaussianBlur,
            "Edge": Edge,
        }
        opts["children"] = [self.operations[n]() for n in opts["default_children"]]
        opts["type"] = "group"
        opts["addText"] = "Add"
        opts["addList"] = [k for k in self.operations.keys()]
        opts["master"] = self
        # setting = self.operations.keys() stores the dict_keys object in this param
        # makes it unpickle-able and thus bugging out before
        super().__init__(**opts)

        for c in self.children():
            c.swap_filter.connect(self.on_swap)

    def on_swap(self, name, direction):
        for i, child in enumerate(self.children()):
            if child.name() == name:
                if direction == "Move Up" and i > 0:
                    self.insertChild(i - 1, child)
                    return
                elif direction == "Move Down" and i < len(self.children()) - 1:
                    self.insertChild(i + 1, child)
                    return

    def addNew(self, typ):
        filter = self.operations[typ]()
        self.addChild(filter, autoIncrementName=True)
        filter.swap_filter.connect(self.on_swap)


class ProcessingGroup(GroupParameter):
    def __init__(self, url, **opts):
        # opts["name"] = "Filters"
        self.operations = {"Bubbles": AnalyzeBubbles}
        opts["children"] = [self.operations[n](url) for n in opts["default_children"]]
        opts["type"] = "group"
        opts["addText"] = "Add"
        opts["addList"] = [k for k in self.operations.keys()]
        super().__init__(**opts)

    def addNew(self, typ):
        filter = self.operations[typ]()
        self.addChild(filter, autoIncrementName=True)


class GeneralSettings(GroupParameter):
    def __init__(self, **opts):
        # opts["name"] = "Settings"

        opts["children"] = [
            FileParameter(name="File Select", value=opts["url"]),
            SliderParameter(name="Overlay Weight", value=1, step=0.01, limits=(0, 1)),
        ]
        super().__init__(**opts)
        self.child("File Select").sigValueChanged.connect(self.file_selected)

    def file_selected(self):
        print(self.child("File Select").value())


class MyParams(ParameterTree):
    paramChange = pyqtSignal(object, object)

    def __init__(self, url=None):
        super().__init__()
        self.name = "MyParams"
        self.settings = QSettings("Bubble Deposition", self.name)
        params = [
            GeneralSettings(name="Settings", url=url),
            FilterGroup(name="Filter", default_children=["Threshold"]),
            ProcessingGroup(name="Analyze", default_children=["Bubbles"], url=url),
        ]

        self.params = Parameter.create(name=self.name, type="group", children=params)

        # load saved data when available or otherwise specified in config.py
        if not RESET_DEFAULT_PARAMS:
            self.state = self.settings.value("State")
            self.params.restoreState(self.state)
            # self.params = pickle.loads(self.state)
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
        # self.state = pickle.dumps(self.params)
        self.state = self.params.saveState()
        pprint(self.state)
        self.settings.setValue("State", self.state)

    def print(self):
        print(self.name)
        print(self.params)
