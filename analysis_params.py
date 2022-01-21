from matplotlib.pyplot import gray, xscale
from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal, Qt

import cv2
import numpy as np
import math

### my classes ###
from bubble_analysis import *
from filters import my_dilate, my_threshold, my_invert
from misc_methods import MyFrame, register_my_param
'''
Analysis Params: Any new analysis methods could be easily implemented here
with its associating parameters. 
def analyze(frame): called in the video/image processing thread and the frame is passed in for processing
- does not need to return anything. state data of the operation is retained here.
def annotate(frame): also in the video thread, called on last after all analysis to draw on annotations. 
- returns annotated
'''


class Analysis(Parameter):

    def __init__(self, **opts):
        opts['removable'] = True
        super().__init__(**opts)
        # self.sigRemoved.connect(self.on_removed)

    def __repr__(self):
        msg = self.opts['name'] + ' Analysis'
        for c in self.childs:
            msg += f'\n{c.name()}: {c.value()}'
        return msg

# Threshold analysis method
# @register_my_param
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
            frame=frame, min_area=self.child('Min Size').value())
        if len(self.bubbles) > self.child('Num Neighbors').value():
            self.lower_bound, self.upper_bound = get_bounds(
                bubbles=self.bubbles,
                scale_x=self.child('Bounds Scale X').value(),
                scale_y=self.child('Bounds Scale Y').value(),
                offset_x=self.child('Bounds Offset X').value(),
                offset_y=self.child('Bounds Offset Y').value(),
            )
            set_neighbors(bubbles=self.bubbles,
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


# algorithm for separating bubbles
# https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa2bfbebbc5c320526897996aafa1d8eb
# - Distance Tranform types
@register_my_param
class Watershed(Parameter):
    cls_type = 'Watershed'

    def __init__(self, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type

        self.img = {
            'orig': None,
            'thresh': None,
            'bg': None,
            'fg': None,
            'dist': None,
            'unknown': None,
            'final': None,
        }

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'Watershed'
        if 'children' not in opts:
            opts['children'] = [{
                'name': 'Toggle',
                'type': 'bool',
                'value': True
            }, {
                'title': 'Min Area',
                'name': 'min_area',
                'type': 'slider',
                'value': 30,
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
                'title': 'D.T. Mask Size',
                'name': 'mask_size',
                'type': 'list',
                'value': 5,
                'limits': [0, 3, 5],
            }, {
                'title': 'D.T. type',
                'name': 'dist_type',
                'type': 'list',
                'value': cv2.DIST_L1,
                'limits': [
                    cv2.DIST_L1, 
                    cv2.DIST_L2, 
                    cv2.DIST_C],
            }, {
                'title': 'View List',
                'name': 'view_list',
                'type': 'list',
                'value': list(self.img.keys())[-1],
                'limits': list(self.img.keys()),
            }]

        super().__init__(**opts)

        self.manual_fg_pts = []
        self.manual_fg_changed = False
        self.manual_fg_size = 1

    def get_frame(self):
        return self.img[self.child('view_list').value()]

    def set_manual_sure_fg(self, pos):
        self.manual_fg_pts.append(pos)
        self.manual_fg_changed = True

    def clear_manual_sure_fg(self):
        self.manual_fg_pts = []

    def watershed_bubble_label(self, frame):
        self.img['orig'] = frame.copy()
        # self.img['thresh'] = my_threshold(self.img['orig'],
        #                                   self.child('Lower').value(),
        #                                   self.child('Upper').value(),
        #                                   'inv thresh')
        self.img['thresh'] = self.img['orig'].cvt_color('gray')
        # expanded threshold to indicate outer bounds of interest
        self.img['bg'] = my_dilate(self.img['thresh'],
                                   iterations=self.child('bg_iter').value())
        # Use distance transform then threshold to find points
        # within the bounds that could be used as seed
        # for watershed
        self.img['dist'] = cv2.distanceTransform(
            self.img['thresh'],             
            self.child('dist_type').value(),
            self.child('mask_size').value())
        # division creates floats, can't have that inside opencv frames
        img_max = np.amax(self.img['dist'])
        if img_max > 0:
            self.img['dist'] = self.img['dist'] * 255 / img_max
        self.img['dist'] = MyFrame(np.uint8(self.img['dist']), 'gray')
        self.img['fg'] = my_threshold(frame=self.img['dist'],
                                      thresh=int(
                                          self.child('fg_scale').value() *
                                          self.img['dist'].max()),
                                      maxval=255,
                                      type='thresh')
        # draw manually selected fg
        if self.manual_fg_changed:
            for pt in self.manual_fg_pts:
                self.img['fg'] = MyFrame(
                    cv2.circle(self.img['fg'], pt, self.manual_fg_size,
                               (255, 255, 255), -1), 'gray')
        self.img['unknown'] = MyFrame(
            cv2.subtract(self.img['bg'], self.img['fg']), 'gray')

        # Marker labeling
        # Labels connected components from 0 - n
        # 0 is for background
        count, markers = cv2.connectedComponents(self.img['fg'])
        markers = MyFrame(markers, 'gray')
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[self.img['unknown'] == 255] = 0
        markers = cv2.watershed(frame.cvt_color('bgr'), markers)
        # border is -1
        # 0 does not exist
        # bg is 1
        # bubbles is >1
        return get_bubbles_from_labels(markers,
                                       min_area=self.child('min_area').value())


@register_my_param
class AnalyzeBubblesWatershed(Analysis):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = 'BubbleAnalysis'

    VIEWING = 0
    EDITING = 1
    DELETE = 2
    UPDATE = 3

    def __init__(self, url, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type
        self.url = url

        self.img = {
            'orig': None,
            'thresh': None,
            'bg': None,
            'fg': None,
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
                'title': 'Num Neighbors',
                'name': 'num_neighbors',
                'type': 'int',
                'value': 3,
                'limits': (1, 255),
            }, {
                'name': 'Conversion',
                'type': 'float',
                'units': 'um/px',
                'value': 600 / 900,
                'readonly': True,
            }, {
                'title': 'Export CSV',
                'name': 'export_csv',
                'type': 'action'
            }, {
                'title': "Export Graphs",
                'name': 'export_graphs',
                'type': 'action'
            }, {
                'title': 'Algorithm',
                'name': 'algo_list',
                'type': 'list',
                'value': 'Watershed',
                'limits': ['Watershed']
            }, {
                'title': 'Watershed Segmentation',
                'name': 'watershed',
                'type': 'Watershed'
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
                ]
            }]
        super().__init__(**opts)

        self.child('export_csv').sigActivated.connect(self.export_csv)
        self.child('export_graphs').sigActivated.connect(self.export_graphs)
        self.sigTreeStateChanged.connect(self.on_param_change)
        # manual sel states:
        self.cursor_x = 0
        self.cursor_y = 0

        self.um_per_pixel = self.child('Conversion').value()

        if 'bubbles' not in opts:
            self.opts['bubbles'] = []

        self.curr_mode = self.VIEWING
        self.prev_mode = self.curr_mode
        self.curr_bubble = None
        self.ignore_auto_labeling = False

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, path):
        self._url = path
        self.ignore_auto_labeling = False

    def on_roi_updated(self, roi):
        print('param update roi')
        self.child('watershed').clear_manual_sure_fg()
        self.ignore_auto_labeling = False

    def on_action_clicked(self, param):
        if param.name() == 'export_csv':
            self.export_csv_flag = True
        elif param.name() == 'export_graphs':
            self.export_graphs_flag = True
        self.ignore_auto_labeling = True

    def on_param_change(self, parameter, changes):
        # self.ignore_auto_labeling = False
        for param, change, data in changes:
            # print(f'{param.name()=}, {change=}, {data=}')
            if param.parent().name(
            ) == 'watershed':  # when watershed algo is called
                self.ignore_auto_labeling = False
            elif change == 'parent':  # called when this parameter is created
                self.ignore_auto_labeling = False
            else:
                self.ignore_auto_labeling = True

    # video thread
    def analyze(self, frame):
        # don't extract bubbles on new frames when not needed
        # eg. moving mouse cursor
        if not self.ignore_auto_labeling:
            # process frame and extract the bubbles with the given algorithm
            self.opts['bubbles'] = self.child(
                'watershed').watershed_bubble_label(frame)
        else:
            self.ignore_auto_labeling = False

        # in the process of adding new bubble
        if self.curr_mode == self.EDITING and self.curr_bubble is not None:
            r = math.dist((self.curr_bubble.x, self.curr_bubble.y),
                          (self.cursor_x, self.cursor_y))
            self.curr_bubble.diameter = r * 2
        # just finished adding new bubble
        elif self.curr_mode == self.VIEWING and self.curr_bubble is not None:
            self.opts['bubbles'].append(self.curr_bubble)
            self.curr_bubble = None
        # deleting bubble
        elif self.curr_mode == self.DELETE:
            self.curr_bubble = None
            b = self.select_bubble(self.cursor_x, self.cursor_y,
                                   self.opts['bubbles'])
            if b is not None:
                self.opts['bubbles'].remove(b)
            self.curr_mode = self.VIEWING

        if len(self.opts['bubbles']) > self.child('num_neighbors').value():
            set_neighbors(self.opts['bubbles'],
                          self.child('num_neighbors').value())

        # return unannotated frame for processing
        # no need
        return frame

    # video thread
    def annotate(self, frame):
        # get current frame selection from the algorithm
        # if not initialized yet, choose standard frame
        view_frame = self.child('watershed').get_frame()
        if view_frame is not None:
            frame = view_frame.cvt_color('bgr')
        else:
            frame = frame.cvt_color('bgr')
        edge_color = (255, 1, 1)
        neighbor_color = (100, 1, 50)
        highlight_color = (1, 1, 120)

        # draw bubble that is being manually drawn
        if self.curr_bubble is not None:
            cv2.circle(frame,
                       (int(self.curr_bubble.x), int(self.curr_bubble.y)),
                       int(self.curr_bubble.diameter / 2), edge_color, 1)

        # highlight bubble under cursor with fill
        if self.child('watershed', 'view_list').value() != 'fg':
            sel_bubble = self.select_bubble(self.cursor_x, self.cursor_y,
                                            self.opts['bubbles'])
        else:
            sel_bubble = None
            
        if sel_bubble is not None:
            cv2.circle(frame, (int(sel_bubble.x), int(sel_bubble.y)),
                       int(sel_bubble.diameter / 2), highlight_color, -1)
            if sel_bubble.neighbors is not None:
                for n in sel_bubble.neighbors:
                    cv2.circle(frame, (int(n.x), int(n.y)),
                               int(n.diameter / 2), neighbor_color, -1)

        # highlight edge of all bubbles
        for b in self.opts['bubbles']:
            cv2.circle(frame, (int(b.x), int(b.y)), int(b.diameter / 2),
                       edge_color, 1)

        # view = self.child('Overlay', 'view_list').value()
        return frame

    def select_bubble(self, x, y, bubbles):
        sel_bubble = None
        for b in bubbles:
            # if cursor within the bubble
            if math.dist((x, y), (b.x, b.y)) < b.diameter / 2:
                if sel_bubble is None:
                    sel_bubble = b
                else:
                    if math.dist((x, y), (b.x, b.y)) < math.dist(
                        (x, y), (sel_bubble.x, sel_bubble.y)):
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
                # create new manual bubble
                if self.child('watershed', 'view_list').value() == 'fg':
                    self.child('watershed').set_manual_sure_fg(
                        pos=(self.cursor_x, self.cursor_y))
                    self.ignore_auto_labeling = False
                else:
                    self.curr_bubble = Bubble(self.cursor_x,
                                              self.cursor_y,
                                              diameter=0,
                                              id=len(self.opts['bubbles']),
                                              type='manual')
                    self.curr_mode = self.EDITING
            elif self.curr_mode == self.EDITING:
                self.curr_mode = self.VIEWING
        elif event.button() == Qt.RightButton:
            self.curr_mode = self.DELETE

    def export_csv(self):
        # print('Export', change)
        self.ignore_auto_labeling = True

        if self.opts['bubbles'] is not None:
            if self.url is None:
                export_csv(  # from bubble_processes
                    bubbles=self.opts['bubbles'],
                    conversion=self.um_per_pixel,
                    url='exported_data',
                )
                print('Default Export')
            else:
                export_csv(
                    bubbles=self.opts['bubbles'],
                    conversion=self.um_per_pixel,
                    url=self.url + '_data',
                )

    def export_graphs(self):
        self.ignore_auto_labeling = True
        print(self.url)
        export_boxplots(
            self.opts['bubbles'],
            self.child('num_neighbors').value(),
            self.um_per_pixel,
            self.url,
        )
        export_scatter(
            self.opts['bubbles'],
            self.child('num_neighbors').value(),
            self.um_per_pixel,
            self.url,
        )
        export_dist_histogram(self.opts['bubbles'],
                              self.child('num_neighbors').value(),
                              self.um_per_pixel, self.url)
        export_diam_histogram(self.opts['bubbles'],
                              self.child('num_neighbors').value(),
                              self.um_per_pixel, self.url)
