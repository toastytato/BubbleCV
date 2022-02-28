from pyqtgraph.parametertree.parameterTypes import *
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.ParameterTree import ParameterTree
from PyQt5.QtCore import QObject, QSettings, pyqtSignal
from pprint import *

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

analysis_types = {
    "BubbleLaser": AnalyzeBubbleLaser,
    "BubblesWatershed": AnalyzeBubblesWatershed
}


@register_my_param
class FilterGroup(GroupParameter):
    cls_type = "FilterGroup"

    def __init__(self, **opts):
        # opts["type"] = "FilterGroup"
        opts["addText"] = "Add"
        opts["addList"] = [k for k in filter_types.keys()]

        names = [c.name() for c in opts['children']]
        names.insert(0, 'Original')

        opts['children'].insert(
            0, {
                'title': 'Frame Preview',
                'name': 'view_list',
                'type': 'list',
                'value': names[0],
                'limits': names,
                'tip': 'Choose which transitionary frame to view for debugging'
            })
        super().__init__(**opts)

        # for c in self.children():
        #     c.swap_filter.connect(self.on_swap)
        # self.sigChildAdded.connect(self.on_child_added)
        self.preview_frame = None

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


@register_my_param
class AnalysisGroup(GroupParameter):
    cls_type = "AnalysisGroup"

    def __init__(self, **opts):
        # opts["name"] = "Filters"
        if "url" not in opts:
            opts["url"] = ""
        opts["addText"] = "Add"
        opts["addList"] = [k for k in analysis_types.keys()]
        super().__init__(**opts)

    def addNew(self, typ):
        operation = analysis_types[typ](url=self.opts["url"])
        self.addChild(operation, autoIncrementName=True)


@register_my_param
class GeneralSettings(GroupParameter):
    cls_type = "SettingsGroup"

    file_sel_signal = pyqtSignal(object)
    roi_clicked_signal = pyqtSignal()

    def __init__(self, **opts):
        if "url" not in opts:
            opts["url"] = opts["default_url"]
        opts["children"] = [
            FileParameter(name="File Select", value=opts["url"]),
            {
                'name': 'curr_frame_idx',
                'title': 'Curr Frame',
                'type': 'int',
                'value': 0,
            },
            # {
            #   'name': 'start_frame_idx',
            #   'title': 'Start Frame',
            #   'type': 'int',
            #   'value': 0,
            # },
            # {
            #   'name': 'end_frame_idx',
            #   'title': 'End Frame',
            #   'type': 'int',
            #   'value': 100,
            # },
            SliderParameter(name="Overlay Weight",
                            value=.9,
                            step=0.01,
                            limits=(0, 1)),
            {
                "name": "Select ROI",
                "type": "action"
            },
        ]
        super().__init__(**opts)
        self.child("File Select").sigValueChanged.connect(self.file_selected)
        self.child("Select ROI").sigActivated.connect(self.roi_clicked)

    def file_selected(self):
        # print(self.child("File Select").value())
        self.file_sel_signal.emit(self.child("File Select").value())

    def roi_clicked(self, change):
        self.roi_clicked_signal.emit()


class MyParams(ParameterTree):
    paramChange = pyqtSignal(object, object)
    paramRestored = pyqtSignal()

    def __init__(self, default_url=None):
        super().__init__()
        self.name = "MyParams"
        self.my_settings = QSettings("Bubble Deposition", self.name)
        params = [
            {
                "name": "Settings",
                "type": "SettingsGroup",
                "default_url": default_url,
            },
            {
                "name":
                "Analyze",
                "type":
                "AnalysisGroup",
                "children": [
                    # AnalyzeBubbleLaser(url=default_url),
                    AnalyzeBubblesWatershed(url=default_url),
                ],
            }
        ]
        self.params = Parameter.create(name=self.name,
                                       type="group",
                                       children=params)

        self.internal_params = {"ROI": [], "Bubbles": []}

        self.restore_settings()
        self.setParameters(self.params, showTop=False)
        self.update_url(self.get_param_value("Settings", "File Select"))
        self.connect_changes()

    def update_url(self, url):
        for p in self.params.child("Analyze").children():
            p.url = url
            print(f'updating {p=} url')

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
            self.internal_params = self.my_settings.value("Internal")
            self.state = self.my_settings.value("State")
            self.params.restoreState(self.state, removeChildren=False)
            self.paramRestored.emit()
            print('restore')

    def save_settings(self):
        self.state = self.params.saveState()
        self.my_settings.setValue("State", self.state)
        self.my_settings.setValue("Internal", self.internal_params)

    def __repr__(self):
        return self.name + "\n" + str(self.params)
