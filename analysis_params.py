from matplotlib.pyplot import gray, xscale
from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal, Qt

import cv2
import numpy as np
import math

### my classes ###
from bubble_contour import *
from filters import my_dilate, my_threshold, my_invert
from misc_methods import MyFrame, register_my_param


class Analysis(Parameter):
    def __init__(self, **opts):
        opts['removable'] = True
        super().__init__(**opts)
        # self.sigRemoved.connect(self.on_removed)

    def analyze(self, frame):
        return frame

    def annotate(self, frame):
        return frame

    def __repr__(self):
        msg = self.opts['name'] + ' Analysis'
        for c in self.childs:
            msg += f'\n{c.name()}: {c.value()}'
        return msg


@register_my_param
class AnalyzeBubbles(Analysis):
    cls_type = 'Bubbles'

    def __init__(self, url, **opts):
        opts['url'] = url
        if 'name' not in opts:
            opts['name'] = 'Bubbles'
        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': True
                },
                {
                    'name': 'Min Size',
                    'type': 'slider',
                    'value': 50,
                    'limits': (0, 200),
                },
                {
                    'name': 'Num Neighbors',
                    'type': 'int',
                    'value': 4
                },
                {
                    'name': 'Bounds Offset X',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Bounds Offset Y',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Bounds Scale X',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Bounds Scale Y',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Conversion',
                    'type': 'float',
                    'units': 'um/px',
                    'value': 600 / 900,
                    'readonly': True,
                },
                {
                    'name': 'Export Distances',
                    'type': 'action'
                },
                {
                    'name': 'Export Graph',
                    'type': 'action'
                },
                {
                    'name':
                    'Overlay',
                    'type':
                    'group',
                    'children': [
                        {
                            'name': 'Toggle',
                            'type': 'bool',
                            'value': True
                        },
                        {
                            'name': 'Bubble Highlight',
                            'type': 'int',
                            'value': 0
                        },
                        {
                            'name': 'Center Color',
                            'type': 'color',
                            'value': '#ff0000',
                        },
                        {
                            'name': 'Circumference Color',
                            'type': 'color',
                            'value': '#2CE2EE',
                        },
                        {
                            'name': 'Neighbor Color',
                            'type': 'color',
                            'value': '#2C22EE',
                        },
                    ],
                },
            ]
        super().__init__(**opts)
        self.bubbles = []
        self.url = url
        self.um_per_pixel = self.child('Conversion').value()
        self.child('Export Distances').sigActivated.connect(self.export_csv)
        self.child('Export Graph').sigActivated.connect(self.export_graphs)

        # self.sigTreeStateChanged.connect(self.on_change)

    def export_csv(self, change):
        # print('Export', change)

        if self.bubbles is not None:
            if self.url is None:
                export_csv(  # from bubble_processes
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url='exported_data',
                )
                print('Default Export')
            else:
                export_csv(
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url=self.url + '_data',
                )

    def export_graphs(self, change):
        print(self.url)
        export_boxplots(
            self.bubbles,
            self.child('Num Neighbors').value(),
            self.um_per_pixel,
            self.url,
        )
        export_scatter(
            self.bubbles,
            self.child('Num Neighbors').value(),
            self.um_per_pixel,
            self.url,
        )
        export_dist_histogram(self.bubbles,
                              self.child('Num Neighbors').value(),
                              self.um_per_pixel, self.url)
        export_diam_histogram(self.bubbles,
                              self.child('Num Neighbors').value(),
                              self.um_per_pixel, self.url)

    def analyze(self, frame):
        self.bubbles = get_bubbles_from_threshold(
            frame=frame, min=self.child('Min Size').value())
        if len(self.bubbles) > self.child('Num Neighbors').value():
            self.lower_bound, self.upper_bound = get_bounds(
                bubbles=self.bubbles,
                scale_x=self.child('Bounds Scale X').value(),
                scale_y=self.child('Bounds Scale Y').value(),
                offset_x=self.child('Bounds Offset X').value(),
                offset_y=self.child('Bounds Offset Y').value(),
            )
            get_neighbors(bubbles=self.bubbles,
                          num_neighbors=self.child('Num Neighbors').value()
                          )  # modifies param to assign neighbors to bubbles
        return frame

    def annotate(self, frame):
        try:
            return draw_annotations(
                frame=frame.cvt_color('bgr'),
                bubbles=self.bubbles,
                min=self.lower_bound,
                max=self.upper_bound,
                highlight_idx=self.child('Overlay',
                                         'Bubble Highlight').value(),
                circum_color=self.child('Overlay',
                                        'Circumference Color').value(),
                center_color=self.child('Overlay', 'Center Color').value(),
                neighbor_color=self.child('Overlay', 'Neighbor Color').value(),
            )
        except AttributeError:
            return frame


@register_my_param
class AnalyzeBubblesWatershed(Analysis):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = 'BubblesWatershed'

    VIEWING = 0
    EDITING = 1
    DELETE = 2

    def __init__(self, url, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type
        opts['url'] = url

        self.img = {
            'gray': None,
            'thresh': None,
            'bg': None,
            'fg': None,
            'sel': None,
            'dist': None,
            'unknown': None,
            'final': None,
        }

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'BubbleWatershed'
        if 'children' not in opts:
            opts['children'] = [{
                'name': 'Toggle',
                'type': 'bool',
                'value': True
            }, {
                'name': 'Upper',
                'type': 'slider',
                'value': 255,
                'limits': (0, 255)
            }, {
                'name': 'Lower',
                'type': 'slider',
                'value': 0,
                'limits': (0, 255)
            }, {
                'title': 'FG scale',
                'name': 'fg_scale',
                'type': 'slider',
                'value': 0.01,
                'precision': 4,
                'step': 0.0005,
                'limits': (0, 1),
            }, {
                'title': 'Adapt Size',
                'name': 'adaptive',
                'type': 'slider',
                'value': 1,
                'step': 1,
                'limits': (1, 255),
            }, {
                'title': 'BG Iterations',
                'name': 'bg_iter',
                'type': 'int',
                'value': 3,
                'limits': (0, 255),
            }, {
                'title': 'Dist Transform Iter.',
                'name': 'dist_iter',
                'type': 'list',
                'value': 5,
                'limits': [0, 3, 5],
            }, {
                'name':
                'Overlay',
                'type':
                'group',
                'children': [
                    {
                        'name': 'Toggle',
                        'type': 'bool',
                        'value': True
                    },
                    {
                        'name': 'Toggle Text',
                        'type': 'bool',
                        'value': False
                    },
                    {
                        'title': 'View List',
                        'name': 'view_list',
                        'type': 'list',
                        'value': list(self.img.keys())[-1],
                        'limits': list(self.img.keys()),
                    },
                ]
            }]
        super().__init__(**opts)

        # manual sel states:
        self.cursor_x = 0
        self.cursor_y = 0

        self.bubbles = []

        self.curr_mode = self.VIEWING
        self.prev_mode = self.curr_mode
        self.curr_bubble = None
        self.ignore_auto_labeling = False

    # video thread
    def analyze(self, frame):
        if not self.ignore_auto_labeling:
            self.bubbles = self.watershed_bubble_label(frame)
        else:
            self.ignore_auto_labeling = False

        # in the process of adding new bubble
        if self.curr_mode == self.EDITING and self.curr_bubble is not None:
            r = math.dist((self.curr_bubble.x, self.curr_bubble.y),
                          (self.cursor_x, self.cursor_y))
            self.curr_bubble.diameter = r * 2
        # just finished adding new bubble
        elif self.curr_mode == self.VIEWING and self.curr_bubble is not None:
            self.bubbles.append(self.curr_bubble)
            self.curr_bubble = None
        # deleting bubble
        elif self.curr_mode == self.DELETE:
            self.curr_bubble = None
            b = self.select_bubble(self.cursor_x, self.cursor_y, self.bubbles)
            if b is not None:
                self.bubbles.remove(b)
            self.curr_mode = self.VIEWING
        # return unannotated frame for processing
        return frame
    
    def watershed_bubble_label(self, frame):
        self.img['gray'] = frame.cvt_color('gray')
        self.img['thresh'] = my_threshold(self.img['gray'],
                                          self.child('Lower').value(),
                                          self.child('Upper').value(),
                                          'inv thresh')
        # expanded threshold to indicate outer bounds of interest
        self.img['bg'] = my_dilate(self.img['thresh'],
                                   iterations=self.child('bg_iter').value())
        # Use distance transform then threshold to find points
        # within the bounds that could be used as seed
        # for watershed
        self.img['dist'] = cv2.distanceTransform(
            self.img['thresh'], cv2.DIST_L2,
            self.child('dist_iter').value())

        # division creates floats, can't have that inside opencv frames
        img_max = np.amax(self.img['dist'])
        if img_max != 0:
            self.img['dist'] = self.img['dist'] * 255 / img_max
        self.img['dist'] = MyFrame(np.uint8(self.img['dist']), 'gray')
        self.img['fg'] = my_threshold(frame=self.img['dist'],
                                      thresh=int(
                                          self.child('fg_scale').value() *
                                          self.img['dist'].max()),
                                      maxval=255,
                                      type='thresh')
        self.img['unknown'] = MyFrame(
            cv2.subtract(self.img['bg'], self.img['fg']), 'gray')

        # Marker labeling
        # Labels connected components from 0 - n
        # 0 is for background
        count, markers = cv2.connectedComponents(self.img['fg'])
        markers = MyFrame(markers, 'gray')
        # print('cc ret:', count)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[self.img['unknown'] == 255] = 0
        # print('Water:', frame.colorspace)
        markers = cv2.watershed(frame.cvt_color('bgr'), markers)
        # border is -1
        # 0 does not exist
        # bg is 1
        # bubbles is >1
        return get_bubbles_from_labels(markers)

    # video thread
    def annotate(self, frame):
        self.img['final'] = frame.cvt_color('bgr')
        edge_color = (255, 0, 0)
        highlight_color = (0, 0, 50)

        # draw bubble that is being manually drawn
        if self.curr_bubble is not None:
            cv2.circle(self.img['final'],
                       (int(self.curr_bubble.x), int(self.curr_bubble.y)),
                       int(self.curr_bubble.diameter / 2), edge_color, 1)

        # highlight bubble under cursor with fill
        sel_bubble = self.select_bubble(self.cursor_x, self.cursor_y, self.bubbles)
        if sel_bubble is not None:
            cv2.circle(self.img['final'],
                       (int(sel_bubble.x), int(sel_bubble.y)),
                       int(sel_bubble.diameter / 2), highlight_color, -1)

        # highlight edge of all bubbles
        for b in self.bubbles:
            cv2.circle(self.img['final'], (int(b.x), int(b.y)),
                       int(b.diameter / 2), edge_color, 1)

        view = self.child('Overlay', 'view_list').value()
        return self.img[view]

    def select_bubble(self, x, y, bubbles):
        sel_bubble = None
        for b in bubbles:
            # if cursor within the bubble
            if math.dist((self.cursor_x, self.cursor_y), (b.x, b.y)) < b.diameter / 2:
                if sel_bubble is None:
                    sel_bubble = b
                else:
                    if math.dist((self.cursor_x, self.cursor_y), (b.x, b.y)) < math.dist(
                        (self.cursor_x, self.cursor_y), (sel_bubble.x, sel_bubble.y)):
                        sel_bubble = b
        return sel_bubble

    def on_mouse_move_event(self, x, y):
        self.ignore_auto_labeling = True
        self.cursor_x = x
        self.cursor_y = y

    def on_mouse_click_event(self, event):
        self.ignore_auto_labeling = True
        if event.button() == Qt.LeftButton:
            if self.curr_mode == self.VIEWING:
                self.curr_bubble = Bubble(self.cursor_x, self.cursor_y, 0, type='manual')
                self.curr_mode = self.EDITING
            elif self.curr_mode == self.EDITING:
                self.curr_mode = self.VIEWING
        elif event.button() == Qt.RightButton:
            self.curr_mode = self.DELETE
