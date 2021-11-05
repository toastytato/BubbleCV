from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

### my classes ###
from my_cv_process import *


class Process(Parameter, QObject):
    finished = pyqtSignal()

    def __init__(self, **opts):
        opts["removable"] = True
        super().__init__(**opts)

    def __repr__(self):
        msg = self.opts["name"] + " Process"
        for c in self.childs:
            msg += f"\n{c.name()}: {c.value()}"
        return msg


class AnalyzeBubbles(Process):
    def __init__(self, **opts):
        opts["name"] = "Bubbles"
        opts["children"] = [
            {"name": "Toggle", "type": "bool", "value": True},
            {"name": "Min Size", "type": "float", "value": 50},
            {"name": "Num Neighbors", "type": "int", "value": 4},
            SliderParameter(
                name="Bounds Offset",
                value=0,
                step=1,
                limits=(-100, 100),
            ),
            {"name": "Export Distances", "type": "action"},
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
                ],
            },
        ]
        super().__init__(**opts)
        self.bubbles = []
        self.sigTreeStateChanged.connect(self.on_change)

    def on_change(self, param, changes):
        for param, change, data in changes:
            if param.name() == "Export Distances":
                if self.bubbles is not None:
                    export_csv(self.bubbles, "exported")

    def process(self, frame):
        self.bubbles = get_contours(frame=frame, min=self.child("Min Size").value())
        if len(self.bubbles) > self.child("Num Neighbors").value():
            self.lower_bound, self.upper_bound = get_bounds(
                bubbles=self.bubbles, offset=self.child("Bounds Offset").value()
            )
            get_neighbors(
                bubbles=self.bubbles, num_neighbors=self.child("Num Neighbors").value()
            )  # modifies param to assign neighbors to bubbles
        return frame

    def annotate(self, frame):
        try:
            return draw_annotations(
                frame=frame,
                bubbles=self.bubbles,
                min=self.lower_bound,
                max=self.upper_bound,
                highlight_idx=self.child("Overlay", "Bubble Highlight").value(),
                circum_color=self.child("Overlay", "Circumference Color").value(),
                center_color=self.child("Overlay", "Center Color").value(),
                neighbor_color=self.child("Overlay", "Neighbor Color").value(),
            )
        except AttributeError:
            return frame

    # calling self.var here might cause slowdown in thread
    def overlay(self, frame):
        pass
