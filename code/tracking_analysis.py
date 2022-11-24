import cv2
import numpy as np
from pyqtgraph.parametertree import Parameter
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed

### my classes ###
from bubble_helpers import (BubbleSubAnalysis, Bubble, get_markers_from_label,
                            get_save_dir, export_all_bubbles_excel,
                            export_imgs)
# from filters import my_dilate, my_erode, my_invert, my_threshold, canny_edge
from filter_params import *
from misc_methods import MyFrame, register_my_param
import main_params as mp
from analysis_params import Analysis

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
            # 'thresh': None,
            # 'bg': None,
            'seed': None,
            # 'eroded': None,
            'dist': None,
            # 'unknown': None,
            'water': None,
            # 'final': None,
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
                3,
                'limits': (1, 255),
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
                'title': 'Debug View',
                'name': 'debug_view',
                'type': 'bool',
                'value': True
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
        frame = self.img[self.child('view_list').value()]
        if frame is None:
            return None
        else:
            return MyFrame(frame)

    def set_manual_sure_fg(self, pos):
        self.manual_fg_pts.append(pos)
        self.manual_fg_changed = True

    def clear_manual_sure_fg(self):
        print('cleared')
        self.manual_fg_pts = []

    # params: frame, kd_tree for temporal coherence with nearest neighbor algo
    # kd_tree is of the PREVIOUS frame's bubbles
    def watershed_get_labels(self, frame, bubbles):

        # self.area_aware_dist(frame)

        self.img['orig'] = frame.copy()
        # self.img['thresh'] = my_threshold(self.img['orig'],
        #                                   self.child('Lower').value(),
        #                                   self.child('Upper').value(),
        #                                   'inv thresh')
        # self.img['thresh'] = self.img['orig'].cvt_color('gray')
        # expanded threshold to indicate outer bounds of interest
        # self.img['bg'] = my_dilate(self.img['thresh'],
        #                            iterations=self.child('bg_iter').value())
        # # Use distance transform then threshold to find points
        # # within the bounds that could be used as seed
        # # for watershed
        # self.img['dist'] = cv2.distanceTransform(
        #     self.img['thresh'],
        #     self.child('dist_type').value(),
        #     self.child('mask_size').value())
        # # division creates floats, can't have that inside opencv frames
        # img_max = np.amax(self.img['dist'])
        # if img_max > 0:
        #     self.img['dist'] = self.img['dist'] * 255 / img_max
        # self.img['dist'] = MyFrame(np.uint8(self.img['dist']), 'gray')

        # # basically doing a erosion operation, but
        # # using the brightness values to erode
        # self.img['seed'] = my_threshold(frame=self.img['dist'],
        #                                 thresh=int(
        #                                     self.child('fg_scale').value() *
        #                                     self.img['dist'].max()),
        #                                 maxval=255,
        #                                 type='thresh')
        # self.img['eroded'] = MyFrame(
        #     cv2.erode(self.img['thresh'], np.ones((10, 10))))
        # coords = peak_local_max(self.img['dist'],
        #                         footprint=np.ones((100, 100)),
        #                         labels=self.img['eroded'])
        # mask = np.zeros(self.img['dist'].shape, dtype=np.uint8)
        # mask[tuple(coords.T)] = 255

        # print(mask.shape)
        # cv2.imshow('mask', mask)

        # markers, _ = ndi.label(mask)
        # self.img['seed'] = MyFrame(mask)
        # print(self.img['seed'].shape)
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

        # self.img['unknown'] = MyFrame(
        #     cv2.subtract(self.img['bg'], self.img['seed']), 'gray')

        # # Marker labeling
        # # Labels connected components from 0 - n
        # # 0 is for background
        # count, markers = cv2.connectedComponents(self.img['seed'])
        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1
        # # Now, mark the region of unknown with zero
        # # delineating the range where the boundary could be
        # markers[self.img['unknown'] == 255] = 0
        # markers = np.uint8(
        #     cv2.watershed(self.img['thresh'].cvt_color('bgr'), markers))

        # self.img['water'] = MyFrame(markers, 'gray')
        # # border is -1
        # # 0 does not exist
        # # bg is 1
        # # bubbles is >1
        self.img['dist'] = ndi.distance_transform_edt(self.img['orig'])
        fp = self.child('min_area').value()
        # take the peak brightness in the distance transforms as that would be around the center of the bubble
        coords = peak_local_max(self.img['dist'],
                                footprint=np.ones((fp, fp)),
                                labels=self.img['orig'])
        self.img['seed'] = np.zeros(self.img['dist'].shape, dtype=np.uint8)
        self.img['seed'][tuple(coords.T)] = True
        # for seeds that are too close to each other, merge them
        self.img['seed'] = cv2.dilate(self.img['seed'],
                                      kernel=None,
                                      iterations=1)
        markers, _ = ndi.label(self.img['seed'])
        self.img['water'] = watershed(-self.img['dist'],
                                      markers,
                                      mask=self.img['orig'])
        return self.img['water']


@register_my_param
class AnalyzeBubblesWatershed(Analysis):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = 'BubbleAnalysis'

    VIEWING = 0
    ADDING = 1
    DELETE = 2
    UPDATE = 3

    def __init__(self, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type

        # becomes class variable once passed into super()
        init_m_type = 'bubbles'
        self.markers = {
            'lasers':
            BubbleSubAnalysis(bubbles=[],
                              frame_idx=550,
                              filters=[
                                  Blur(),
                                  Threshold(lower=180, type='thresh'),
                              ]),
            'bubbles':
            BubbleSubAnalysis(
                bubbles=[],
                frame_idx=0,
                filters=[
                    #   Blur(),
                    Normalize(),
                    Blur(),
                    Threshold(type='otsu'),
                    Invert(),
                    Dilate(),
                    Erode()
                ]),
            'deposits':
            BubbleSubAnalysis(bubbles=[],
                              frame_idx=11436,
                              filters=[
                                  Blur(),
                                  Contrast(brightness=55, contrast=1.4),
                                  Threshold(lower=148, type='thresh')
                              ]),
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

        self.child('filter_group').filters_updated_signal.connect(
            self.update_curr_marker_filters)

        # using opts so that it can be saved
        # not sure if necessary
        # when I can just recompute bubbles
        self.annotated_frame = np.array([])
        self.prev_rec_state = False
        self.curr_mode = self.VIEWING
        self.prev_mode = self.curr_mode
        self.curr_bubble = None
        self.new_bubble_init_pos = 0

    @property
    def m_type(self):
        return self.child('analysis_group', 'marker_list').value()

    @m_type.setter
    def m_type(self, marker):
        self.child('analysis_group', 'marker_list').setValue(marker)

    def update_curr_marker_filters(self, filters):
        self.markers[self.m_type].filters = filters

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

        export_all_bubbles_excel(bubbles=b_interest,
                                 lasers=l_interest,
                                 deposits=d_interest,
                                 roi=self.opts['roi'],
                                 framerate=self.rec_framerate,
                                 conversion=self.um_per_pixel,
                                 url=self.opts['url'])

        # # only export if there is data
        # if len(b_interest) and len(l_interest) and len(l_interest):
        #     export_all_bubbles_excel(bubbles=b_interest,
        #                              lasers=l_interest,
        #                              deposits=d_interest,
        #                              roi=self.opts['roi'],
        #                              framerate=self.rec_framerate,
        #                              conversion=self.um_per_pixel,
        #                              url=self.opts['url'])
        # else:
        #     print('There is an empty dataset')

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
                reference_marker = 'bubbles'
                self.markers[reference_marker].correlate_other_ids_to_self(
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
                # self.request_annotate_update.emit()
                self.request_analysis_update.emit()

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
        else:
            pass
        # associate the neighboring bubbles
        # num_neigbors = self.child('num_neighbors').value()
        # associate each bubble to its # nearest neighbors
        # set_neighbors(self.marker_kd_trees[self.marker_type], num_neigbors)

    # called in video thread
    # every update to the frame/imageview
    def annotate(self, frame):
        # get current frame selection from the algorithm
        # if not initialized yet, choose standard frame
        if self.child('analysis_group', 'watershed', 'debug_view').value():
            view_frame = self.child('analysis_group', 'watershed').get_frame()
        else:
            view_frame = self.crop_to_roi(
                self.child('filter_group').get_preview())

        if view_frame is not None:
            # don't crop because the images stored in the analysis are already cropped
            frame = view_frame.cvt_color('bgr')
        else:
            frame = self.crop_to_roi(frame.cvt_color('bgr'))
        # print(frame)
        # full scale range so that we can see dim objects
        diff = frame.max() - frame.min()
        if diff > 0:
            p = 256 / diff
        else:
            p = 1
        a = frame.min() * p
        # print(f'p:{p}, a:{a}, min:{frame.min()}, max:{frame.max()}')

        frame = p * frame - a

        if not self.child("Overlay", "Toggle").value():
            self.annotated_frame = frame.copy()
            return frame

        # isolate by replacing frame with gray background
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

        if self.curr_mode == self.ADDING:
            center = np.add(self.cursor_pos, self.new_bubble_init_pos) / 2
            radius = np.linalg.norm(
                np.subtract(self.cursor_pos, self.new_bubble_init_pos) / 2)

            self.markers[self.m_type].manual_bubble.update(
                x=center[0], y=center[1], r=radius, frame=self.curr_frame_idx)

        for b in self.markers[self.m_type].bubbles['curr']:
            if self.m_type == 'bubbles':
                if b.state == Bubble.SELECTED:
                    sel_color = (0, 255, 0)
                    cv2.circle(frame, b.ipos, int(b.r), sel_color, 1)
                else:
                    cv2.circle(frame, b.ipos, int(b.r), edge_color, 1)

            if self.child('Overlay', 'Toggle Center').value():
                if b.state == Bubble.SELECTED:
                    cv2.circle(frame, b.ipos, 2, highlight_color, -1)
                elif b.state == Bubble.MANUAL:
                    cv2.circle(frame, b.ipos, 1, (40, 200, 200), -1)
                else:
                    cv2.circle(frame, b.ipos, 2, (0, 0, 255), -1)

            if self.child('Overlay', 'Toggle Text').value():
                cv2.putText(frame, str(b.id), (int(b.x) - 11, int(b.y) + 7),
                            cv2.FONT_HERSHEY_PLAIN, 1, text_color)

        if self.child('Overlay', 'Toggle Text').value():
            (h, w) = frame.shape[:2]
            text_pos = (int(w - 400), int(h - 40))
            legend_pos = (int(w - 400), int(h - 80))
            cv2.putText(frame, 'Analyzing: ' + self.m_type, text_pos,
                        cv2.FONT_HERSHEY_PLAIN, 2, text_color)
            num_um_ref = 20
            cv2.putText(frame, f'{num_um_ref} um:', legend_pos,
                        cv2.FONT_HERSHEY_PLAIN, 1, text_color)
            ref_length_px = int(num_um_ref / self.um_per_pixel)

            cv2.line(frame, (w - 315, h - 83),
                     (w - 315 + ref_length_px, h - 83), text_color, 2)

        self.annotated_frame = frame.copy()

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

        # don't highlight bubble while it is being added
        if (curr_sel_bubble is not None and self.m_type == 'bubbles'
                and self.curr_mode != self.ADDING):
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

    def on_keypress_event(self, key):
        if key == ord(' '):
            print('deleting')
            if self.curr_mode == self.VIEWING:
                b = self.markers[self.m_type].get_bubble_containing_pt(
                    self.cursor_pos)
                if b is not None:
                    self.markers[self.m_type].remove_bubble(b)
            elif self.curr_mode == self.ADDING:
                b = self.markers[self.m_type].manual_bubble
                if b is not None:
                    self.markers[self.m_type].remove_bubble(b)
                self.curr_mode = self.VIEWING
        super().on_keypress_event(key)

    def on_mouse_move_event(self, x, y):
        super().on_mouse_move_event(x, y)

    def on_mouse_click_event(self, event):
        if event == 'left':
            if self.m_type == 'deposits':
                self.markers[self.m_type].create_manual_bubble(
                    pos=self.cursor_pos,
                    r=10,  # give it a radius so it's easier to recognize
                    frame_idx=self.curr_frame_idx,
                    state=Bubble.MANUAL)
            # there is a bubble under mouse cursor
            elif self.markers[self.m_type].toggle_state_bubble_containing_pt(
                    self.cursor_pos) and self.curr_mode == self.VIEWING:
                print('toggled bubble state')
            elif self.m_type == 'bubbles':
                if self.curr_mode == self.VIEWING:
                    print('creating')
                    self.new_bubble_init_pos = self.cursor_pos
                    self.markers[self.m_type].create_manual_bubble(
                        pos=self.new_bubble_init_pos,
                        r=0,
                        frame_idx=self.curr_frame_idx,
                        state=Bubble.MANUAL)
                    self.curr_mode = self.ADDING
                elif self.curr_mode == self.ADDING:
                    if self.markers[self.m_type].manual_bubble is not None:
                        self.markers[
                            self.m_type].manual_bubble.state = Bubble.SELECTED
                    self.curr_mode = self.VIEWING
        super().on_mouse_click_event(event)


class BubbleShiftAnalysis(Analysis):

    def __init__(self, **opts) -> None:

        if 'name' not in opts:
            opts['name'] = 'BubbleShift'

        if 'children' not in opts:
            opts['children'] = [{
                'title': 'Preprocessing',
                'name': 'filter_group',
                'type': 'FilterGroup',
                'expanded': False,
                'children': [Blur()]
            }, {
                'title':
                'Analysis Params',
                'name':
                'analysis_group',
                'type':
                'group',
                'children': [{
                    'title': 'Frame Select',
                    'name': 'frame_sel',
                    'type': 'list',
                    'value': 'prev',
                    'limits': ['prev', 'next']
                }]
            }]

        super().__init__(**opts)

        self.frame = {
            'prev': {
                'frame': None,
                'idx': 0
            },
            'next': {
                'frame': None,
                'idx': 0
            },
        }

    @property
    def f_type(self):
        return self.child('analysis_group', 'frame_sel')

    @f_type.setter
    def f_type(self, value):
        self.child('analysis_group', 'frame_sel').setValue(value)

    def analyze(self, frame):
        self.frame.cvt_color('gray')
        super().analyze(frame)

    def annotate(self, frame):
        return super().annotate(frame)
