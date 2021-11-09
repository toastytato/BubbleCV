from os import DirEntry
from pyqtgraph.parametertree.parameterTypes import *
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.ParameterTree import ParameterTree
from PyQt5.QtCore import QObject, QSettings, pyqtSignal
from pprint import *

### my classes ###
from filters import *
from filter_params import *
from processing_params import *

RESET_DEFAULT_PARAMS = True

# keys are the names in the Add list
# update this as new filters are added
# keys does not represent actual name of the Parameter type
# Parameter type is stored in cls_type
filter_types = {
    "Threshold": Threshold,
    "Dilate": Dilate,
    "Erode": Erode,
    "Invert": Invert,
    "Gaussian Blur": GaussianBlur,
    "Edge": Edge,
}

operation_types = {
    "Bubbles": AnalyzeBubbles,
}


class FilterGroup(GroupParameter):
    def __init__(self, **opts):
        # opts["type"] = "FilterGroup"
        opts["addText"] = "Add"
        opts["addList"] = [k for k in filter_types.keys()]

        super().__init__(**opts)

        for c in self.children():
            c.swap_filter.connect(self.on_swap)
        self.sigChildAdded.connect(self.on_child_added)

    def on_swap(self, name, direction):
        for i, child in enumerate(self.children()):
            if child.name() == name:
                if direction == "Move Up" and i > 0:
                    self.insertChild(i - 1, child)
                    return
                elif direction == "Move Down" and i < len(self.children()) - 1:
                    self.insertChild(i + 1, child)
                    return

    # either new child is created or child is restored
    def on_child_added(self, parent, child, index):
        child.swap_filter.connect(self.on_swap)
        # print("Adding", child.name(), ":", child.getValues())

    # when the Add filter is clicked
    def addNew(self, typ):
        filter = filter_types[typ]()  # create new filter
        self.addChild(filter, autoIncrementName=True)  # emits sigChildAdded


class ProcessingGroup(GroupParameter):
    def __init__(self, **opts):
        # opts["name"] = "Filters"
        opts["addText"] = "Add"
        opts["addList"] = [k for k in operation_types.keys()]
        super().__init__(**opts)

    def addNew(self, typ):
        operation = operation_types[typ]()
        self.addChild(operation, autoIncrementName=True)


class GeneralSettings(GroupParameter):
    fileSelectedSignal = pyqtSignal(object)

    def __init__(self, **opts):
        opts["children"] = [
            FileParameter(name="File Select", value=opts["url"]),
            SliderParameter(name="Overlay Weight", value=1, step=0.01, limits=(0, 1)),
        ]
        super().__init__(**opts)
        self.child("File Select").sigValueChanged.connect(self.file_selected)

    def file_selected(self):
        # print(self.child("File Select").value())
        self.fileSelectedSignal.emit(self.child("File Select").value())


# ---- Register custom parameter types ----
print("Registering Custom Group Parameters")
registerParameterType("FilterGroup", FilterGroup)
registerParameterType("ProcessingGroup", ProcessingGroup)
registerParameterType("SettingsGroup", GeneralSettings)
for cls in filter_types.values():
    registerParameterType(cls.cls_type, cls)
for cls in operation_types.values():
    registerParameterType(cls.cls_type, cls)


class MyParams(ParameterTree):
    paramChange = pyqtSignal(object, object)

    def __init__(self, url=None):
        super().__init__()
        self.name = "MyParams"
        self.settings = QSettings("Bubble Deposition", self.name)
        params = [
            {
                "name": "Settings",
                "type": "SettingsGroup",
                "url": url,
            },
            {
                "name": "Filter",
                "type": "FilterGroup",
                # "children": [Threshold()],    # starting default filters
            },
            {
                "name": "Analyze",
                "type": "ProcessingGroup",
                "children": [AnalyzeBubbles(url)],
            }
            # ProcessingGroup(name="Analyze", children=["Bubbles"], url=url),
        ]
        self.params = Parameter.create(name=self.name, type="group", children=params)

        self.restore_settings()

        self.setParameters(self.params, showTop=False)
        self.params.sigTreeStateChanged.connect(self.send_change)

    def send_change(self, param, changes):
        self.paramChange.emit(param, changes)

    # Convienience methods for modifying parameter values.
    def get_param_value(self, *childs):
        """Get the current value of a parameter."""
        return self.params.child(*childs).value()

    def set_param_value(self, value, *childs):
        """Set the current value of a parameter."""
        return self.params.child(*childs).setValue(value)

    def restore_settings(self):
        # load saved data when available or otherwise specified in config.py
        if not RESET_DEFAULT_PARAMS:
            self.state = self.settings.value("State")
            self.params.restoreState(self.state)

    def save_settings(self):
        self.state = self.params.saveState()
        self.settings.setValue("State", self.state)

    def __repr__(self):
        return self.name + "\n" + str(self.params)
