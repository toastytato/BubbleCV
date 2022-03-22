from atexit import register
from inspect import FrameInfo
import math
from re import S
from tkinter import Frame

import cv2
import numpy as np
from cv2 import FONT_HERSHEY_PLAIN, erode
from matplotlib import scale
from matplotlib.pyplot import gray, xscale
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from pygments import highlight
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import SliderParameter, FileParameter

### my classes ###
from bubble_helpers import (BubblesGroup, Bubble, get_markers_from_label,
                            get_save_dir, export_all_bubbles_excel,
                            export_imgs)
from filters import my_dilate, my_erode, my_invert, my_threshold
from filter_params import *
from misc_methods import MyFrame, register_my_param
import main_params as mp
'''
Analysis Params: Any new analysis methods could be easily implemented here
with its associating parameters. 
def analyze(frame): called in the video/image processing thread and the frame is passed in for processing
- does not need to return anything. state data of the operation is retained here.
def annotate(frame): also in the video thread, called on last after all analysis to draw on annotations. 
- returns annotated
'''


# boiler plate code for analysis params
class Analysis(Parameter):

    # params: analyze: bool, annotate: bool
    # tells the main controller whether or not
    # to update analysis/annotations
    request_analysis_update = pyqtSignal()
    request_annotate_update = pyqtSignal()
    # true for play, false for pause
    request_resume = pyqtSignal(bool)
    # set curr frame
    request_set_frame_idx = pyqtSignal(int)
    # update video url
    request_url_update = pyqtSignal(str)

    def __init__(self, **opts):
        opts['removable'] = True
        if 'name' not in opts:
            opts['name'] = 'DefaultAnalysis'

        if 'children' not in opts:
            opts['children'] = []

        all_children_names = [c['name'] for c in opts['children']]

        opts['children'].insert(
            0, {
                'name':
                'Settings',
                'type':
                'group',
                'children': [
                    FileParameter(name="File Select", value=opts['url']),
                    {
                        'name': 'curr_frame_idx',
                        'title': 'Curr Frame',
                        'type': 'int',
                        'value': opts.get('curr_frame_idx', 0),
                    },
                    {
                        'title': 'Play',
                        "name": "playback",
                        "type": "action"
                    },
                    {
                        "title": "Select ROI",
                        'name': 'sel_roi',
                        "type": "action"
                    },
                ]
            })

        if 'Overlay' not in all_children_names:
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

        # print(self.opts)

        # for c in self.opts['children']:
        #     print('c:', c)
        #     if c['type'] != 'group':
        #         setattr(self, f"get_{c['name']}", lambda: c.value())
        # self.sigRemoved.connect(self.on_removed)'

        self.child('Settings', 'sel_roi').sigActivated.connect(self.set_roi)
        self.child('Settings', 'curr_frame_idx').sigValueChanged.connect(
            self.send_frame_idx_request)
        self.child('Settings',
                   'playback').sigActivated.connect(self.toggle_playback)
        self.child('Settings', 'File Select').sigValueChanged.connect(
            lambda x: self.request_url_update.emit(x.value()))

        # manual sel states:
        self.opts['roi'] = None
        self.orig_frame = np.array([])
        self.cursor_pos = [0, 0]
        self.is_playing = False

    # put algorithmic operations here
    # called on parameter updates
    def analyze(self, frame):
        self.orig_frame = frame

    # put lighter operations
    # in here which will be called on every update
    def annotate(self, frame):
        return frame

    def send_frame_idx_request(self, param):
        self.request_set_frame_idx.emit(int(param.value()))

    @property
    def curr_frame_idx(self):
        return self.child('Settings', 'curr_frame_idx').value()

    # will tell video thread to update to current frame
    @curr_frame_idx.setter
    def curr_frame_idx(self, idx):
        param = self.child('Settings', 'curr_frame_idx')
        param.setValue(idx)

    # will NOT tell video thread to update to current frame
    # only updates view
    # prevent recursive call to itself due to signal being triggered
    def set_curr_frame_idx_no_emit(self, idx):
        param = self.child('Settings', 'curr_frame_idx')
        param.sigValueChanged.disconnect(self.send_frame_idx_request)
        param.setValue(idx)
        param.sigValueChanged.connect(self.send_frame_idx_request)

    @property
    def is_playing(self):
        # shows pause while playing and vice versa
        return (self.child('Settings', 'playback').title() == 'Pause')

    # True: is playing
    @is_playing.setter
    def is_playing(self, resume):
        p = self.child('Settings', 'playback')
        if resume:
            p.setOpts(title='Pause')
            self.child('analysis_group', 'watershed', 'Toggle').setValue(True)
        else:
            p.setOpts(title='Play')

    def toggle_playback(self):
        self.is_playing = not self.is_playing

    def crop_to_roi(self, frame):
        if self.opts['roi'] is not None:
            return frame[int(self.opts['roi'][1]):int(self.opts['roi'][1] +
                                                      self.opts['roi'][3]),
                         int(self.opts['roi'][0]):int(self.opts['roi'][0] +
                                                      self.opts['roi'][2])]
        else:
            (h, w) = frame.shape
            self.opts['roi'] = [0, 0, w, h]
            return frame

    def set_roi(self):
        r = cv2.selectROI("Select ROI", self.orig_frame)
        if all(r) != 0:
            self.opts['roi'] = r
        cv2.destroyWindow("Select ROI")
        self.request_analysis_update.emit()

    def on_mouse_click_event(self, event):
        print('click', event)
        self.request_annotate_update.emit()

    def on_mouse_move_event(self, x, y):
        self.cursor_pos = [x, y]
        self.request_annotate_update.emit()

    def __repr__(self):
        msg = self.opts['name'] + ' Analysis'
        for c in self.childs:
            msg += f'\n{c.name()}: {c.value()}'
        return msg


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
                2,
                'limits': (0, 255),
                'tip':
                'Bubble area below this value is considered noise and ignored'
            }, {
                'title': 'FG scale',
                'name': 'fg_scale',
                'type': 'slider',
                'value': 0.01,
                'precision': 4,
                'step': 0.01,
                'limits': (0, 1),
                'visible': True
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
        self.img['seed'] = my_threshold(frame=self.img['dist'],
                                        thresh=int(
                                            self.child('fg_scale').value() *
                                            self.img['dist'].max()),
                                        maxval=255,
                                        type='thresh')
        # self.img['seed'] = self.area_aware_erosion(
        #     frame=self.img['thresh'],
        #     scaling=self.child('fg_scale').value(),
        #     iterations=self.child('erode_iters').value())

        # # self.img['final_fg'] = np.zeros(self.img['seed'].shape, dtype=np.uint8)
        # # draw manually selected fg
        # if self.manual_fg_changed:
        #     for pt in self.manual_fg_pts:
        #         self.img['seed'] = MyFrame(
        #             cv2.circle(self.img['seed'], pt, self.manual_fg_size,
        #                        (255, 255, 255), -1), 'gray')

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
class AnalyzeBubblesWatershed(Analysis):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = 'BubbleAnalysis'

    VIEWING = 0
    SELECTING = 1
    DELETE = 2
    UPDATE = 3

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

        # becomes class variable once passed into super()
        init_m_type = 'bubbles'
        self.markers = {
            'lasers':
            BubblesGroup(bubbles=[],
                         frame_idx=550,
                         filters=[
                             Blur(),
                             Threshold(lower=180, type='thresh'),
                         ]),
            'bubbles':
            BubblesGroup(bubbles=[],
                         frame_idx=970,
                         filters=[
                             Blur(),
                             Normalize(),
                             Blur(name='Blur2'),
                             Threshold(lower=45, type='inv thresh'),
                         ]),
            'deposits':
            BubblesGroup(bubbles=[],
                         frame_idx=11436,
                         filters=[
                             Blur(),
                             Contrast(brightness=55, contrast=1.4),
                             Threshold(lower=148, type='thresh')
                         ])
        }

        opts['curr_frame_idx'] = self.markers[init_m_type].frame_idx

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'BubbleWatershed'

        if 'children' not in opts:
            opts['children'] = [{
                'title': 'Preprocessing',
                'name': 'filter_group',
                'type': 'FilterGroup',
                'expanded': False,
                'children': self.markers[init_m_type].filters
            }, {
                'title':
                'Analysis Params',
                'name':
                'analysis_group',
                'type':
                'group',
                'children': [
                    {
                        'title': 'Num Neighbors',
                        'name': 'num_neighbors',
                        'type': 'int',
                        'value': 3,
                        'limits': (1, 255),
                        'visible': False
                    },
                    {
                        'title': 'Marker View',
                        'name': 'marker_list',
                        'type': 'list',
                        'value': list(self.markers.keys())[1],
                        'limits': list(self.markers.keys())
                    },
                    {
                        'title': 'Analyze`',
                        'name': 'reanalyze',
                        'type': 'action',
                    },
                    {
                        'title': 'Correlate Curr ID to Bubble ID',
                        'name': 'correlate_id',
                        'type': 'action'
                    },
                    {
                        'title': 'Mass Select',
                        'name': 'mass_sel',
                        'type': 'action'
                    },
                    {
                        'title': 'Clear Markers',
                        'name': 'reset_markers',
                        'type': 'action',
                    },
                    {
                        'title': 'Watershed Segmentation',
                        'name': 'watershed',
                        'type': 'Watershed'
                    },
                ]
            }, {
                'title':
                'Export Params',
                'name':
                'export_group',
                'type':
                'group',
                'children': [
                    {
                        'name': 'Conversion',
                        'type': 'float',
                        'units': 'um/px',
                        'value': 600 / 1280,
                        'readonly': True,
                    },
                    {
                        'title': 'Recorded Framerate',
                        'name': 'rec_fps',
                        'type': 'float',
                        'units': 'fps',
                        'value': 30,
                        'readonly': True,
                    },
                    {
                        'name': 'toggle_rec',
                        'title': 'Toggle Recording',
                        'type': 'bool',
                        'value': False
                    },
                    {
                        'name': 'end_frame',
                        'title': 'End Rec Frame',
                        'type': 'int',
                        'value': 10500
                    },
                    {
                        'title': 'Export Curr Frame',
                        'name': 'export_frame',
                        'type': 'action'
                    },
                    {
                        'title': 'Export Data',
                        'name': 'export_csv',
                        'type': 'action'
                    },
                    {
                        'title': 'Export Graphs',
                        'name': 'export_graphs',
                        'type': 'action',
                        'visible': False
                    },
                ]
            }, {
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
                    'value': True
                }, {
                    'name': 'Toggle Center',
                    'type': 'bool',
                    'value': True
                }, {
                    'name': 'Isolate Markers',
                    'type': 'bool',
                    'value': False
                }]
            }]
        super().__init__(**opts)

        self.sigTreeStateChanged.connect(self.on_param_change)

        # print(self.get_end_frame())

        self.um_per_pixel = self.child('export_group', 'Conversion').value()
        self.rec_framerate = self.child('export_group', 'rec_fps').value()

        # using opts so that it can be saved
        # not sure if necessary
        # when I can just recompute bubbles
        self.annotated_frame = np.array([])
        self.prev_rec_state = False
        self.curr_mode = self.VIEWING
        self.prev_mode = self.curr_mode
        self.curr_bubble = None

    @property
    def m_type(self):
        return self.child('analysis_group', 'marker_list').value()

    @m_type.setter
    def m_type(self, marker):
        self.child('analysis_group', 'marker_list').setValue(marker)

    # overwrite analysis set_roi in order to reset markers
    def set_roi(self):
        r = cv2.selectROI("Select ROI", self.orig_frame)
        if all(r) != 0:
            self.opts['roi'] = r
        self.reset_markers()
        self.request_analysis_update.emit()
        cv2.destroyWindow("Select ROI")

    def reset_markers(self):
        print('marker reset')
        self.child('analysis_group', 'watershed').clear_manual_sure_fg()
        self.markers[self.m_type].clear()
        self.request_annotate_update.emit()

    def mass_select(self):
        title = "Select bubbles of interest"
        bubble_roi = cv2.selectROI(title, self.annotated_frame)
        cv2.destroyWindow(title)
        self.markers[self.m_type].set_state_bubble_in_roi(
            bubble_roi, Bubble.SELECTED)
        self.request_annotate_update.emit()

    def export_data(self):
        b_interest = self.markers['bubbles'].get_bubbles_of_state(
            state=Bubble.SELECTED, type='all')
        l_interest = self.markers['lasers'].get_bubbles_of_state(
            state=Bubble.SELECTED, type='all')
        d_interest = self.markers['deposits'].get_bubbles_of_state(
            state=Bubble.SELECTED, type='all')
        # export_imgs(frame=self.orig_frame,
        #             bubbles=b_interest,
        #             lasers=l_interest,
        #             deposits=d_interest)

        # only export if there is data
        if len(b_interest) and len(l_interest) and len(l_interest):
            export_all_bubbles_excel(bubbles=b_interest,
                                     lasers=l_interest,
                                     deposits=d_interest,
                                     roi=self.opts['roi'],
                                     framerate=self.rec_framerate,
                                     conversion=self.um_per_pixel,
                                     url=self.opts['url'])
        else:
            print('There is an empty dataset')

    # meant to disable or enable re analyze
    # cuz sometimes frame hasn't updated, just the param values
    # but some param value updates do not require recalculating
    # frame values
    def on_param_change(self, parameter, changes):
        for param, change, data in changes:
            # print(f'{param.name()=}, {change=}, {data=}')
            name = param.name()

            if name == 'File Select':
                self.setOpts(url=data)
            elif name == 'marker_list':
                self.curr_frame_idx = self.markers[self.m_type].frame_idx
                self.child('analysis_group', 'watershed',
                           'Toggle').setValue(False)
                self.child('filter_group').replace_filters(
                    self.markers[self.m_type].filters)
                self.request_analysis_update.emit()
                return
            elif name == 'export_csv':
                self.export_data()

            elif name == 'export_frame':
                if self.child('Overlay', 'Isolate Markers').value():
                    cv2.imwrite(f'export/{self.m_type}_isolated.png',
                                self.annotated_frame)
                else:
                    cv2.imwrite(f'export/{self.m_type}.png',
                                self.annotated_frame)
            elif name == 'mass_sel':
                # self.mass_select()
                for b in self.markers[self.m_type].bubbles['curr']:
                    b.state = Bubble.SELECTED
            elif name == 'reset_markers':
                self.reset_markers()
            elif name == 'correlate_id':
                self.markers['bubbles'].correlate_other_ids_to_self(
                    other_group=self.markers[self.m_type], type='curr')
            elif name == 'reanalyze':
                self.child('analysis_group', 'watershed',
                           'Toggle').setValue(True)
                self.request_analysis_update.emit()

            parent = param.parent()
            if (isinstance(parent, Filter) or param.name() == 'view_list'):
                # when watershed algo is called
                self.request_analysis_update.emit()
            else:
                self.request_annotate_update.emit()

    # video thread
    def analyze(self, frame):
        frame = frame.cvt_color('gray')
        super().analyze(frame)
        self.markers[self.m_type].frame_idx = self.curr_frame_idx
        self.markers[self.m_type].frame = self.orig_frame
        # preprocessing for the analysis
        frame = self.crop_to_roi(self.child('filter_group').preprocess(frame))

        if self.child('analysis_group', 'watershed', 'Toggle').value():
            # process frame and extract the bubbles with the given algorithm
            # if kd_tree is empty, create IDs from scratch
            # print('analyzing')
            labels = self.child(
                'analysis_group', 'watershed').watershed_get_labels(
                    frame=frame,
                    bubbles=self.markers[self.m_type].bubbles['curr'])
            markers = get_markers_from_label(labeled_frame=labels,
                                             min_area=self.child(
                                                 'analysis_group', 'watershed',
                                                 'min_area').value(),
                                             fit_circle='perimeter')
            self.markers[self.m_type].update_bubbles_to_new_markers(
                markers, self.curr_frame_idx)
        # associate the neighboring bubbles
        # num_neigbors = self.child('num_neighbors').value()
        # associate each bubble to its # nearest neighbors
        # set_neighbors(self.marker_kd_trees[self.marker_type], num_neigbors)

    # called in video thread
    def annotate(self, frame):
        # get current frame selection from the algorithm
        # if not initialized yet, choose standard frame
        view_frame = self.crop_to_roi(self.child('filter_group').get_preview())
        if view_frame is not None:
            frame = view_frame.cvt_color('bgr')
        else:
            frame = frame.cvt_color('bgr')

        if not self.child("Overlay", "Toggle").value():
            return frame

        if self.child('Overlay', 'Isolate Markers').value():
            (h, w) = frame.shape[:2]
            frame = 128 * np.ones((h, w, 3), dtype=np.uint8)

        # cv2.imshow('fr', frame)
        edge_color = (255, 1, 1)
        highlight_color = (50, 200, 50)
        text_color = (255, 255, 255)

        # neighbor_color = (20, 200, 20)
        # if sel_bubble.neighbors is not None:
        #     for n in sel_bubble.neighbors:
        #         if n.state != Bubble.REMOVED:
        #             cv2.circle(frame, n.ipos, n.ir, neighbor_color, -1)
        # highlight edge of all bubbles
        for b in self.markers[self.m_type].bubbles['curr']:
            if self.m_type == 'bubbles':
                if b.state == Bubble.SELECTED:
                    sel_color = (0, 255, 0)
                    cv2.circle(frame, b.ipos, int(b.r), sel_color, 1)
                else:
                    cv2.circle(frame, b.ipos, int(b.r), edge_color, 1)

            if self.child('Overlay', 'Toggle Center').value():
                if b.state == Bubble.SELECTED:
                    cv2.circle(frame, b.ipos, 3, highlight_color, -1)
                elif b.state == Bubble.MANUAL:
                    cv2.circle(frame, b.ipos, 3, (40, 200, 200), -1)
                else:
                    cv2.circle(frame, b.ipos, 3, (0, 0, 255), -1)

            if self.child('Overlay', 'Toggle Text').value():
                cv2.putText(frame, str(b.id), (int(b.x) - 11, int(b.y) + 7),
                            FONT_HERSHEY_PLAIN, 1, text_color)

        if self.child('Overlay', 'Toggle Text').value():
            (h, w) = frame.shape[:2]
            text_pos = (int(w - 400), int(h - 40))
            legend_pos = (int(w - 400), int(h - 80))
            cv2.putText(frame, 'Analyzing: ' + self.m_type, text_pos,
                        FONT_HERSHEY_PLAIN, 2, text_color)
            num_um_ref = 20
            cv2.putText(frame, f'{num_um_ref} um:', legend_pos,
                        FONT_HERSHEY_PLAIN, 1, text_color)
            ref_length_px = int(num_um_ref / self.um_per_pixel)

            cv2.line(frame, (w - 315, h - 83),
                     (w - 315 + ref_length_px, h - 83), text_color, 2)

        self.annotated_frame = frame

        # # if fg selection don't highlight so user can see the dot
        # if self.child('analysis_group', 'watershed',
        #               'view_list').value() != 'seed':
        #     curr_sel_bubble = self.markers[
        #         self.m_type].get_bubble_containing_pt(self.cursor_pos)
        # else:
        #     curr_sel_bubble = None
        # highlight bubble under cursor with fill
        curr_sel_bubble = self.markers[self.m_type].get_bubble_containing_pt(
            self.cursor_pos)
        if curr_sel_bubble is not None:
            cv2.circle(frame, curr_sel_bubble.ipos, curr_sel_bubble.ir,
                       highlight_color, 5)
        self.save_to_video(frame)

        if (self.is_playing and self.curr_frame_idx >= self.child(
                'export_group', 'end_frame').value()):
            self.is_playing = False

        return frame

    def save_to_video(self, frame):
        if self.curr_frame_idx >= self.child(
                'export_group', 'end_frame').value() and self.is_playing:
            self.child('export_group', 'toggle_rec').setValue(False)

        curr_rec_state = self.child('export_group', 'toggle_rec').value()

        if curr_rec_state:
            # rising edge
            if not self.prev_rec_state:
                (h, w) = frame.shape[:2]
                self.vid_writer = cv2.VideoWriter(
                    'export/rec_video.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                    10, (w, h))  # get width and height
            self.vid_writer.write(frame)
        else:
            # falling edge
            if self.prev_rec_state:
                self.vid_writer.release()
                print('requesting pause')
                self.is_playing = False

        self.prev_rec_state = curr_rec_state

    def on_mouse_move_event(self, x, y):
        super().on_mouse_move_event(x, y)

    def on_mouse_click_event(self, event):
        super().on_mouse_click_event(event)
        if self.m_type == 'deposits':
            self.markers[self.m_type].add_manual_bubble(
                pos=self.cursor_pos,
                r=10,  # irrelevant in this scenario
                frame_idx=self.curr_frame_idx,
                state=Bubble.MANUAL)
        else:
            if event == 'left':
                self.markers[self.m_type].set_state_bubble_containing_pt(
                    self.cursor_pos, Bubble.SELECTED)
            elif event == 'right':
                self.markers[self.m_type].set_state_bubble_containing_pt(
                    self.cursor_pos, Bubble.AUTO)
