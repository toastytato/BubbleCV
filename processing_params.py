from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

### my classes ###
from bubble_process import *


class Process(Parameter):
    def __init__(self, **opts):
        opts["removable"] = True
        super().__init__(**opts)

    def process(self, frame):
        return frame

    def annotate(self, frame):
        return frame

    def __repr__(self):
        msg = self.opts["name"] + " Process"
        for c in self.childs:
            msg += f"\n{c.name()}: {c.value()}"
        return msg


class AnalyzeBubbles(Process):
    cls_type = "Bubbles"

    def __init__(self, url, **opts):
        if "name" not in opts:
            opts["name"] = "Bubbles"
        if "children" not in opts:
            opts["children"] = [
                {"name": "Toggle", "type": "bool", "value": True},
                {
                    "name": "Min Size",
                    "type": "slider",
                    "value": 50,
                    "limits": (0, 200),
                },
                {"name": "Num Neighbors", "type": "int", "value": 4},
                {
                    "name": "Bounds Offset X",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Bounds Offset Y",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Bounds Scale X",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Bounds Scale Y",
                    "type": "slider",
                    "value": 0,
                    "step": 1,
                    "limits": (-200, 200),
                },
                {
                    "name": "Conversion",
                    "type": "float",
                    "units": "um/px",
                    "value": 600 / 900,
                    "readonly": True,
                },
                {"name": "Export Distances", "type": "action"},
                {"name": "Export Graph", "type": "action"},
                {
                    "name": "Overlay",
                    "type": "group",
                    "children": [
                        {"name": "Toggle", "type": "bool", "value": True},
                        {"name": "Bubble Highlight", "type": "int", "value": 0},
                        {
                            "name": "Center Color",
                            "type": "color",
                            "value": "#ff0000",
                        },
                        {
                            "name": "Circumference Color",
                            "type": "color",
                            "value": "#2CE2EE",
                        },
                        {
                            "name": "Neighbor Color",
                            "type": "color",
                            "value": "#2C22EE",
                        },
                    ],
                },
            ]
        super().__init__(**opts)
        self.bubbles = []
        self.url = url
        self.um_per_pixel = self.child("Conversion").value()
        self.child("Export Distances").sigActivated.connect(self.export_csv)
        self.child("Export Graph").sigActivated.connect(self.export_graphs)

        # self.sigTreeStateChanged.connect(self.on_change)

    def export_csv(self, change):
        # print("Export", change)

        if self.bubbles is not None:
            if self.url is None:
                export_csv(  # from bubble_processes
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url="exported_data",
                )
                print("Default Export")
            else:
                export_csv(
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url=self.url + "_data",
                )

    def export_graphs(self, change):
        export_boxplots(
            self.bubbles,
            self.child("Num Neighbors").value(),
            self.um_per_pixel,
            self.url,
        )
        export_scatter(
            self.bubbles,
            self.child("Num Neighbors").value(),
            self.um_per_pixel,
            self.url,
        )

    def process(self, frame):
        self.bubbles = get_contours(frame=frame, min=self.child("Min Size").value())
        if len(self.bubbles) > self.child("Num Neighbors").value():
            self.lower_bound, self.upper_bound = get_bounds(
                bubbles=self.bubbles,
                scale_x=self.child("Bounds Scale X").value(),
                scale_y=self.child("Bounds Scale Y").value(),
                offset_x=self.child("Bounds Offset X").value(),
                offset_y=self.child("Bounds Offset Y").value(),
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


class HoughCircles(Process):
    def __init__(self, **opts):
        opts["name"] = "HoughCircles"
        opts["children"] = [
            {"name": "Param1", "type": "int", "value": 100},
            {"name": "Param2", "type": "int", "value": 100},
        ]
