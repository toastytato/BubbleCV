from atexit import register
import math
from tkinter import Frame

import cv2
import numpy as np
from cv2 import FONT_HERSHEY_PLAIN, erode
from matplotlib import scale
from matplotlib.pyplot import gray, xscale
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import SliderParameter

### my classes ###
from bubble_analysis import *
from filters import my_dilate, my_erode, my_invert, my_threshold
from filter_params import *
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

    # params: analyze: bool, annotate: bool
    # tells the main controller whether or not
    # to update analysis/annotations
    request_display_update = pyqtSignal(bool, bool)

    def __init__(self, **opts):
        opts['removable'] = True
        if 'name' not in opts:
            opts['name'] = 'DefaultAnalysis'

        for c in opts['children']:
            if c['name'] == 'Overlay':
                break
        else:
            opts['children'].append({
                'name':
                'Overlay',
                'type':
                'group',
                'children': [{
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': True
                }]
            })
        super().__init__(**opts)
        # self.sigRemoved.connect(self.on_removed)'
        # manual sel states:
        self.cursor_x = 0
        self.cursor_y = 0

    def analyze(self, frame, frame_idx):
        pass

    def annotate(self, frame):
        return frame

    def on_mouse_click_event(self, event):
        self.request_display_update.emit(True, True)

    def on_mouse_move_event(self, x, y):
        self.cursor_x = x
        self.cursor_y = y
        print('request display update')
        self.request_display_update.emit(False, True)

    def on_roi_updated(self, roi):
        print('roi updated')

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
            # modifies param to assign neighbors to bubbles
            set_neighbors(bubbles=self.bubbles,
                          num_neighbors=self.child('Num Neighbors').value())
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
            'seed': None,
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
                'title':
                'Min Area',
                'name':
                'min_area',
                'type':
                'slider',
                'value':
                5,
                'limits': (0, 255),
                'tip':
                'Bubbles with area below this value is considered noise and ignored'
            }, {
                'title': 'FG scale',
                'name': 'fg_scale',
                'type': 'slider',
                'value': 0.01,
                'precision': 4,
                'step': 0.01,
                'limits': (0, 1),
                'visible': False
            }, {
                'title':
                'Seed Erode Iters',
                'name':
                'erode_iters',
                'type':
                'int',
                'value':
                1,
                'step':
                1,
                'limits': (1, 255),
                'tip':
                '# of times to shrink the segmentation seed'
            }, {
                'title': 'BG Dilate Iters',
                'name': 'bg_iter',
                'type': 'int',
                'value': 3,
                'limits': (0, 255),
            }, {
                'title': 'Laplacian Threshold',
                'name': 'lap_thresh',
                'type': 'slider',
                'value': 50,
                'limits': (-127, 128),
                'visible': False,
            }, {
                'title': 'Dist Mask Size',
                'name': 'mask_size',
                'type': 'list',
                'value': 5,
                'limits': [0, 3, 5],
                'visible': False,
            }, {
                'title':
                'D.T. type',
                'name':
                'dist_type',
                'type':
                'list',
                'value':
                cv2.DIST_L1,
                'limits': [cv2.DIST_L1, cv2.DIST_L2, cv2.DIST_C],
            }, {
                'title':
                'View List',
                'name':
                'view_list',
                'type':
                'list',
                'value':
                list(self.img.keys())[-1],
                'limits':
                list(self.img.keys()),
                'tip':
                'Choose which transitionary frame to view for debugging'
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
        print('cleared')
        self.manual_fg_pts = []

    # scale area proportional to how big they are
    def area_aware_dist(self, frame):
        frame = frame.cvt_color('gray')

        dist_trans = cv2.distanceTransform(frame,
                                           self.child('dist_type').value(),
                                           self.child('mask_size').value())
        img_max = np.amax(dist_trans)
        if img_max > 0:
            dist_trans = dist_trans * 255 / img_max
        dist_trans = MyFrame(np.uint8(dist_trans), 'gray')
        blur = cv2.GaussianBlur(dist_trans, (3, 3), 3)

        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        grad_x = cv2.Sobel(blur,
                           ddepth,
                           dx=1,
                           dy=0,
                           ksize=5,
                           scale=scale,
                           delta=delta,
                           borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(blur,
                           ddepth,
                           dx=0,
                           dy=1,
                           ksize=5,
                           scale=scale,
                           delta=delta,
                           borderType=cv2.BORDER_DEFAULT)

        # abs_grad_x = cv2.convertScaleAbs(grad_x)
        # abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

        lapl_x = cv2.Laplacian(blur, cv2.CV_8S, ksize=5)
        print(np.amax(lapl_x))

        # _, lapl_x = cv2.threshold(
        #     grad,
        #     self.child('lap_thresh').value(),
        #     127,
        #     cv2.THRESH_BINARY)
        # lapl_x = cv2.convertScaleAbs(lapl_x)
        mask = np.zeros(lapl_x.shape, np.uint8)
        mask[lapl_x < self.child('lap_thresh').value()] = 255
        cv2.imshow('Deriv on Dist', grad_y)
        cv2.imshow('dist', dist_trans)
        cv2.imshow('lapl', mask)
        # division creates floats, can't have that inside opencv frames

        # count, labeled_frame = cv2.connectedComponents(frame)
        # out = MyFrame(np.zeros(frame.shape, dtype='uint8'))
        # for i in range(self.child('erode_iters').value()):
        #     for label in np.unique(labeled_frame):
        #         mask = np.zeros(labeled_frame.shape, dtype='uint8')
        #         mask[labeled_frame == label] = 255
        #         isolated_dist = cv2.bitwise_and(dist_trans,
        #                                         dist_trans,
        #                                         mask=mask)
        #         isolated_dist = my_threshold(
        #             frame=MyFrame(isolated_dist, 'gray'),
        #             thresh=int(scale_factor * isolated_dist.max()),
        #             maxval=255,
        #             type='thresh')
        #         out += isolated_dist
        #         # detect contours in the mask and grab the largest one
        #     thresh = my_threshold(out, 1, 255, 'thresh')
        #     count, labeled_frame = cv2.connectedComponents(thresh)
        # return out

    def area_aware_erosion(self, frame, scaling, iterations=1):
        gray = frame.cvt_color('gray')

        max_area = 10000
        max_kernel_size = 100
        min_kern_size = 3

        # TODO: very inefficient
        # only proceed if at least one contour was found
        for i in range(iterations):
            out = MyFrame(np.zeros(frame.shape, dtype='uint8'))
            count, labeled_frame = cv2.connectedComponents(gray)
            for label in np.unique(labeled_frame):
                mask = np.zeros(labeled_frame.shape, dtype='uint8')
                mask[labeled_frame == label] = 255
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                if len(cnts) > 1:
                    continue
                # c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(cnts[0])
                if area < max_area:
                    # k = max(
                    #     int(area*scaling*max_kernel_size/max_area),
                    #     min_kern_size)
                    k = 3
                    if area > 30:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (k, k))
                        out += cv2.erode(mask, kernel)
                    else:
                        out += mask
            gray = out
            # detect contours in the mask and grab the largest one
        return out

    # params: frame, kd_tree for temporal coherence with nearest neighbor algo
    # kd_tree is of the PREVIOUS frame's bubbles
    def watershed_get_labels(self, frame, bubbles):

        # self.area_aware_dist(frame)

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

        # basically doing a erosion operation, but
        # using the brightness values to erode
        # self.img['seed'] = my_threshold(
        #     frame=self.img['dist'],
        #     thresh=int(self.child('fg_scale').value() * self.img['dist'].max()),
        #     maxval=255,
        #     type='thresh')
        self.img['seed'] = self.area_aware_erosion(
            frame=self.img['thresh'],
            scaling=self.child('fg_scale').value(),
            iterations=self.child('erode_iters').value())

        # self.img['final_fg'] = np.zeros(self.img['seed'].shape, dtype=np.uint8)
        # draw manually selected fg
        if self.manual_fg_changed:
            for pt in self.manual_fg_pts:
                self.img['seed'] = MyFrame(
                    cv2.circle(self.img['seed'], pt, self.manual_fg_size,
                               (255, 255, 255), -1), 'gray')

        # if a frame has a manual_fg, overlay that onto the 'seed' for one frame
        # to get the bubbles from that manual_fg
        # still calculate auto fg every frame
        # if the auto_fg has a cont_fg on top of it, use the cont_fg
        # else use the auto_fg as that indicates it's a new bubble

        # for b in bubbles:
        #     if b.state == Bubble.REMOVED:
        #         continue
        #     if self.img['seed'][b.y][b.x]:    # auto_fg intersects bubble center
        #         pass
        #     self.img['seed'] = MyFrame(
        #         cv2.circle(self.img['seed'], b.ipos, 1,
        #                     (255, 255, 255), -1), 'gray')

        self.img['unknown'] = MyFrame(
            cv2.subtract(self.img['bg'], self.img['seed']), 'gray')

        # Marker labeling
        # Labels connected components from 0 - n
        # 0 is for background
        count, markers = cv2.connectedComponents(self.img['seed'])
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        # delineating the range where the boundary could be
        markers[self.img['unknown'] == 255] = 0
        markers = cv2.watershed(frame.cvt_color('bgr'), markers)
        # border is -1
        # 0 does not exist
        # bg is 1
        # bubbles is >1
        return MyFrame(markers, 'gray')


@register_my_param
class AnalyzeBubbleLaser(Analysis):
    cls_type = 'LaserAnalysis'

    def __init__(self, url, **opts):

        preprocess_operations = [
            Blur(),
            Threshold(lower=160),
        ]
        preview_names = [p.name() for p in preprocess_operations]
        preview_names.append('final')
        if 'name' not in opts:
            opts['name'] = 'LaserAnalysis'
        if 'children' not in opts:
            opts['children'] = [{
                'name': 'Toggle',
                'type': 'bool',
                'value': True
            }, {
                'title': 'Preprocessing',
                'name': 'filter',
                'type': 'FilterGroup',
                'expanded': False,
                'children': preprocess_operations
            }, {
                'title': 'Set Laser',
                'name': 'set_laser',
                'type': 'action'
            }, {
                'title': 'Set Deposit',
                'name': 'set_deposit',
                'type': 'action'
            }, {
                'title': 'Calculate',
                'name': 'calculate',
                'type': 'action'
            }]
        super().__init__(**opts)

        self.laser_frame = None
        self.laser_markers = []
        self.deposit_frame = None
        self.deposit_markers = []
        self.curr_frame = None
        self.curr_markers = []

        self.prev_frame_idx = 0

        self.child('set_laser').sigActivated.connect(self.on_set_laser)
        self.child('set_deposit').sigActivated.connect(self.on_set_deposit)
        self.child('calculate').sigActivated.connect(self.calculate)

    def on_mouse_click_event(self, event):
        if event.button() == Qt.LeftButton:
            self.deposit_markers.append(
                Bubble(x=self.cursor_x,
                       y=self.cursor_y,
                       r=1,
                       frame=self.curr_frame_idx))
        return super().on_mouse_click_event(event)

    def on_set_laser(self):
        self.laser_frame = self.curr_frame
        self.laser_markers = self.curr_markers

    def on_set_deposit(self):
        self.deposit_frame = self.curr_frame

    def calculate(self):
        pass

    def analyze(self, frame, frame_idx):
        self.curr_frame_idx = frame_idx
        self.curr_frame = MyFrame(self.child('filter').preprocess(frame))

        # clear manual markers on new frame
        # only set manual markers on a single frame
        if frame_idx != self.prev_frame_idx:
            gray = self.curr_frame.cvt_color('gray')
            Bubble.reset_id()
            self.curr_markers = get_bubbles_from_threshold(gray)
            self.deposit_markers = []

        self.prev_frame_idx = frame_idx

    def annotate(self, frame):
        # get current selected preview if it exists
        img = MyFrame(self.child('filter').get_preview())
        if img is not None:
            frame = img

        frame.cvt_color('bgr')

        for m in self.curr_markers:
            frame = cv2.circle(frame, m.ipos, 3, (255, 0, 0), -1)
            text_color = (255, 0, 0)
            cv2.putText(frame, str(m.id), (int(m.x) - 11, int(m.y) + 7),
                        FONT_HERSHEY_PLAIN, 1, text_color)
        for m in self.deposit_markers:
            frame = cv2.circle(frame, m.ipos, 3, (0, 0, 255), -1)
            text_color = (0, 0, 255)
            cv2.putText(frame, str(m.id), (int(m.x) - 11, int(m.y) + 7),
                        FONT_HERSHEY_PLAIN, 1, text_color)

        return frame


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
            'seed': None,
            'dist': None,
            'unknown': None,
            'final': None,
        }

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'BubbleWatershed'
        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': True
                },
                {
                    'title':
                    'Preprocessing',
                    'name':
                    'filter',
                    'type':
                    'FilterGroup',
                    'expanded':
                    False,
                    'children': [
                        Blur(),
                        # Normalize(),
                        Contrast(),
                        Threshold(lower=60),
                        Invert(),
                    ],  # starting default filters
                },
                {
                    'title': 'Num Neighbors',
                    'name': 'num_neighbors',
                    'type': 'int',
                    'value': 3,
                    'limits': (1, 255),
                },
                {
                    'name': 'Conversion',
                    'type': 'float',
                    'units': 'um/px',
                    'value': 600 / 900,
                    'readonly': True,
                },
                {
                    'title': 'Recorded Framerate',
                    'name': 'rec_fps',
                    'type': 'float',
                    'units': 'fps',
                    'value': 100,
                    'readonly': True,
                },
                {
                    'title': 'Export CSV',
                    'name': 'export_csv',
                    'type': 'action'
                },
                {
                    'title': 'Export Graphs',
                    'name': 'export_graphs',
                    'type': 'action',
                    'visible': False
                },
                {
                    'title': 'Reset ID',
                    'name': 'reset_id',
                    'type': 'action',
                },
                {
                    'title': 'Watershed Segmentation',
                    'name': 'watershed',
                    'type': 'Watershed'
                },
                {
                    'name':
                    'Overlay',
                    'type':
                    'group',
                    'children': [{
                        'name': 'Toggle',
                        'type': 'bool',
                        'value': True
                    }, {
                        'name': 'Toggle Text',
                        'type': 'bool',
                        'value': False
                    }]
                }
            ]
        super().__init__(**opts)

        self.child('export_csv').sigActivated.connect(self.export_csv)
        self.child('export_graphs').sigActivated.connect(self.export_graphs)
        self.child('reset_id').sigActivated.connect(self.reset_markers)

        self.sigTreeStateChanged.connect(self.on_param_change)

        self.um_per_pixel = self.child('Conversion').value()
        self.rec_framerate = self.child('rec_fps').value()

        # using opts so that it can be saved
        # not sure if necessary
        # when I can just recompute bubbles
        if 'bubbles' not in opts:
            self.opts['bubbles'] = []

        self.all_bubbles = []

        self.curr_mode = self.VIEWING
        self.prev_mode = self.curr_mode
        self.curr_bubble = None
        self.auto_label = True
        self.bubble_kd_tree = None

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, path):
        self._url = path
        self.auto_label = True

    def reset_markers(self):
        print('marker reset')
        self.child('watershed').clear_manual_sure_fg()
        self.auto_label = True
        self.bubble_kd_tree = None
        self.opts['bubbles'] = []
        self.all_bubbles = []
        Bubble.id_cnt = 0

    def on_roi_updated(self, roi):
        print('param update roi')
        self.reset_markers()

    def on_action_clicked(self, param):
        if param.name() == 'export_csv':
            self.export_csv_flag = True
        elif param.name() == 'export_graphs':
            self.export_graphs_flag = True
        self.auto_label = False

    # meant to disable or enable auto labeling
    # cuz sometimes frame hasn't updated, just the param values
    # but some param value updates do not require recalculating
    # frame values
    def on_param_change(self, parameter, changes):
        # self.auto_label = False
        for param, change, data in changes:
            print(f'{param.name()=}, {change=}, {data=}')
            if param.parent().name() == 'watershed':
                # when watershed algo is called
                self.auto_label = True
            elif change == 'parent':  # called when this parameter is created
                self.auto_label = True

    # video thread
    def analyze(self, frame, frame_idx):
        # preprocessing for the analysis
        print('anals')
        frame = self.child('filter').preprocess(frame)

        # don't extract bubbles on new frames when not needed
        # eg. moving mouse cursor
        if self.auto_label and self.child('watershed', 'Toggle').value():
            # process frame and extract the bubbles with the given algorithm
            # if kd_tree is empty, create IDs from scratch
            labels = self.child('watershed').watershed_get_labels(
                frame=frame, bubbles=self.opts['bubbles'])
            self.bubble_kd_tree = get_bubbles_from_labels(
                labeled_frame=labels,
                frame_idx=frame_idx,
                min_area=self.child('watershed', 'min_area').value(),
                fit_circle='perimeter',
                prev_kd_tree=self.bubble_kd_tree)
            self.opts['bubbles'] = self.bubble_kd_tree.bubbles
        else:
            self.auto_label = True

        # associate the neighboring bubbles
        num_neigbors = self.child('num_neighbors').value()
        # associate each bubble to its # nearest neighbors
        set_neighbors(self.bubble_kd_tree, num_neigbors)

        # return unannotated frame for processing
        # no need
        return frame

    # called in video thread
    def annotate(self, frame):
        # get current frame selection from the algorithm
        # if not initialized yet, choose standard frame
        view_frame = self.child('filter').get_preview()
        if view_frame is not None:
            frame = view_frame.cvt_color('bgr')
        else:
            frame = frame.cvt_color('bgr')

        # cv2.imshow('fr', frame)

        edge_color = (255, 1, 1)
        neighbor_color = (100, 1, 50)
        highlight_color = (1, 1, 120)

        # draw bubble that is being manually drawn
        # if self.curr_bubble is not None:
        # in the process of adding new bubble
        if self.curr_mode == self.EDITING:
            # radius of bubble is from clicked pos to current cursor pos
            self.curr_bubble.r = math.dist(
                (self.curr_bubble.x, self.curr_bubble.y),
                (self.cursor_x, self.cursor_y))
            cv2.circle(frame, self.curr_bubble.ipos, self.curr_bubble.ir,
                       edge_color, 1)
        # just finished adding new bubble
        elif self.curr_mode == self.VIEWING and self.curr_bubble is not None:
            self.opts['bubbles'].append(self.curr_bubble)
            self.curr_bubble = None
        # deleting bubble
        elif self.curr_mode == self.DELETE:
            self.curr_bubble = None
            b = self.select_bubble((self.cursor_x, self.cursor_y),
                                   self.bubble_kd_tree)
            if b is not None:
                b.state = Bubble.REMOVED
            self.curr_mode = self.VIEWING

        # if fg selection don't highlight so user can see the dot
        if self.child('watershed', 'view_list').value() != 'seed':
            sel_bubble = self.select_bubble((self.cursor_x, self.cursor_y),
                                            self.bubble_kd_tree)
        else:
            sel_bubble = None
        # highlight bubble under cursor with fill
        if sel_bubble is not None:
            cv2.circle(frame, sel_bubble.ipos, sel_bubble.ir, highlight_color,
                       -1)
            if sel_bubble.neighbors is not None:
                for n in sel_bubble.neighbors:
                    if n.state != Bubble.REMOVED:
                        cv2.circle(frame, n.ipos, n.ir, neighbor_color, -1)

        # highlight edge of all bubbles
        for b in self.opts['bubbles']:
            if b.state == Bubble.REMOVED:
                continue

            if b not in self.all_bubbles:
                self.all_bubbles.append(b)

            cv2.circle(frame, b.ipos, int(b.r), edge_color, 1)
            if self.child('Overlay', 'Toggle Text').value():
                text_color = (255, 255, 255)
                cv2.putText(frame, str(b.id), (int(b.x) - 11, int(b.y) + 7),
                            FONT_HERSHEY_PLAIN, 1, text_color)

        # view = self.child('Overlay', 'view_list').value()
        return frame

    # get the bubble that contains the point within its boundaries
    def select_bubble(self, point, kd_tree):
        sel_bubble = None
        if kd_tree is None:
            return sel_bubble
        bubbles = kd_tree.bubbles
        # check all the bubbles to see if cursor is inside
        for b in bubbles:
            if b.state == Bubble.REMOVED:
                continue
            # if cursor within the bubble
            if math.dist(point, b.pos) < b.r:
                if sel_bubble is None:
                    sel_bubble = b
                # if cursor within multiple bubbles, select the closer one
                else:
                    if (math.dist(point, b.pos) < math.dist(
                            point, sel_bubble.pos)):
                        sel_bubble = b
        return sel_bubble

        # find bubble closest to the cursor via center
        # dist, nn_b = kd_tree.get_nn_bubble_dist_from_point((x,y))

        # # if cursor is inside of the nearest bubble
        # if dist < nn_b.r:
        #     return nn_b
        # else:
        #     return None

    def on_mouse_move_event(self, x, y):
        super().on_mouse_move_event(x, y)
        # self.auto_label = False

    def on_mouse_click_event(self, event):
        super().on_mouse_click_event(event)
        self.auto_label = True
        if event.button() == Qt.LeftButton:
            if self.curr_mode == self.VIEWING:
                # create new manual bubble
                if self.child('watershed', 'view_list').value() == 'seed':
                    self.child('watershed').set_manual_sure_fg(
                        pos=(self.cursor_x, self.cursor_y))
                    self.auto_label = False
                # else:
                #     self.curr_bubble = Bubble(x=self.cursor_x,
                #                               y=self.cursor_y,
                #                               r=0,
                #                               state=Bubble.MANUAL)
                #     self.curr_mode = self.EDITING
            elif self.curr_mode == self.EDITING:
                self.curr_mode = self.VIEWING
        elif event.button() == Qt.RightButton:
            self.curr_mode = self.DELETE

    def export_csv(self):
        # print('Export', change)
        self.auto_label = False

        # if self.opts['bubbles'] is not None:
        #     if self.url is None:
        #         export_csv(  # from bubble_processes
        #             bubbles=self.opts['bubbles'],
        #             conversion=self.um_per_pixel,
        #             url='exported_data',
        #         )
        #         print('Default Export')
        #     else:
        #         export_csv(
        #             bubbles=self.opts['bubbles'],
        #             conversion=self.um_per_pixel,
        #             url=self.url + '_data',
        #         )

        export_all_bubbles_excel(self.all_bubbles, self.rec_framerate,
                                 self.um_per_pixel)

    def export_graphs(self):
        self.auto_label = False
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