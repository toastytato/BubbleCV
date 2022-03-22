from pyqtgraph.parametertree.parameterTypes import *
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.ParameterTree import ParameterTree
from PyQt5.QtCore import QObject, QSettings, pyqtSignal
from pprint import *
import numpy as np

### my classes ###
from misc_methods import register_my_param
from filters import *
from filter_params import *
from analysis_params import *

RESET_DEFAULT_PARAMS = 1

# keys are the names in the Add list
# update this as new filters are added
# keys does not represent actual name of the Parameter type
# Parameter type is stored in cls_type
filter_types = {
    'Normalize': Normalize,
    "Threshold": Threshold,
    'Contrast': Contrast,
    "Dilate": Dilate,
    "Erode": Erode,
    "Invert": Invert,
    "Blur": Blur,
    "Edge": Edge,
    "Hough": HoughCircle,
}

analysis_types = {"BubblesWatershed": AnalyzeBubblesWatershed}


@register_my_param
class FilterGroup(GroupParameter):
    cls_type = "FilterGroup"

    def __init__(self, **opts):
        # opts["type"] = "FilterGroup"
        opts["addText"] = "Add"
        opts["addList"] = [k for k in filter_types.keys()]

        self.view_limits = [c.name() for c in opts['children']]
        self.view_limits.insert(0, 'Original')

        super().__init__(**opts)

        self.insertChild(
            0, {
                'title': 'Frame Preview',
                'name': 'view_list',
                'type': 'list',
                'value': self.view_limits[0],
                'limits': self.view_limits,
                'tip': 'Choose which transitionary frame to view for debugging'
            })
        # for c in self.children():
        #     c.swap_filter.connect(self.on_swap)
        self.sigChildAdded.connect(self.on_child_added)
        self.sigChildAdded.connect(self.update_limits)
        self.sigChildRemoved.connect(self.update_limits)
        self.preview_frame = np.array([])

    def update_limits(self):
        limits = [c.name() for c in self.children()]
        limits[0] = 'Original'
        self.child('view_list').setLimits(limits)

    def get_filters(self):
        return self.children()[1:]

    def replace_filters(self, filters):
        # don't remove the view list
        for c in self.children()[1:]:
            self.removeChild(c)
        self.addChildren(filters)

    def preprocess(self, frame):
        if self.child('view_list').value() == 'Original':
            self.preview_frame = frame
        for f in self.children():
            if isinstance(f, Filter) and f.child('Toggle').value():
                frame = f.process(frame)
                if self.child('view_list').value() == f.name():
                    self.preview_frame = frame

        return frame

    def get_preview(self):
        # make sure to copy
        # or else annotate will be working of frame instance that was
        # obtained in analysis which will cause
        # bad things to happen
        return self.preview_frame.copy()

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


class MyParams(ParameterTree):
    paramChange = pyqtSignal(object, object)
    paramRestored = pyqtSignal()

    def __init__(self, default_url=None):
        super().__init__()
        self.name = "MyParams"
        self.my_settings = QSettings("Bubble Deposition", self.name)
        params = [
            AnalyzeBubblesWatershed(url=default_url),
        ]
        self.params = Parameter.create(name=self.name,
                                       type="group",
                                       children=params)

        self.restore_settings()
        self.setParameters(self.params, showTop=False)
        # self.update_url(self.get_param_value("Settings", "File Select"))
        self.connect_changes()

    def disconnect_changes(self):
        self.params.sigTreeStateChanged.disconnect(self.send_change)

    def connect_changes(self):
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

    def get_child(self, *childs):
        return self.params.child(*childs)

    def restore_settings(self):
        # load saved data when available or otherwise specified in config.py
        if not RESET_DEFAULT_PARAMS:
            self.state = self.my_settings.value("State")
            self.params.restoreState(self.state, removeChildren=False)
            self.paramRestored.emit()
            print('restore')

    def save_settings(self):
        self.state = self.params.saveState()
        self.my_settings.setValue("State", self.state)

    def __repr__(self):
        return self.name + "\n" + str(self.params)
