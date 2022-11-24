from analysis_params import Analysis
import filter_params as fp
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import numpy as np
import cv2
import imutils
import os
import math
import pandas as pd
import openpyxl

import pyqtgraph.parametertree.parameterTypes as ptypes
from misc_methods import MyFrame, register_my_param


class Bubble:
    id_cnt = 0

    def __init__(self,
                 mask,
                 init_idx,
                 shrink,
                 blur=3,
                 param1=20,
                 param2=10) -> None:
        self.id = Bubble.id_cnt
        Bubble.id_cnt += 1

        self.lifetime = 1
        self.init_idx = init_idx
        self.exists = True

        self.init_mask = cv2.erode(mask, kernel=None, iterations=shrink)
        self.init_mask = cv2.medianBlur(self.init_mask, blur * 2 + 1)
        h, w = mask.shape
        self.x, self.y, self.r = self.get_circle_from_mask(
            self.init_mask, param1, param2)
        self.roi_x1 = max(int(self.x - self.r), 0)
        self.roi_y1 = max(int(self.y - self.r), 0)
        self.roi_x2 = min(int(self.roi_x1 + self.r * 2), w)
        self.roi_y2 = min(int(self.roi_y1 + self.r * 2), h)
        self.cropped_mask = self.init_mask[self.roi_y1:self.roi_y2,
                                           self.roi_x1:self.roi_x2]

    def get_circle_from_mask(self, mask, param1, param2):
        # # detect contours in the mask and grab the largest one
        # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        #                         cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # c = max(cnts, key=cv2.contourArea)
        # area = cv2.contourArea(c)
        # # draw a circle enclosing the object
        # # keep r if using min enclosing circle radius
        # ((x, y), r) = cv2.minEnclosingCircle(c)
        # # get center via Center of Mass
        # M_1 = cv2.moments(c)
        # if M_1['m00'] == 0:
        #     M_1['m00'] = 1
        # x = int((M_1['m10']) / (M_1['m00']))
        # y = int((M_1['m01']) / (M_1['m00']))

        # fit_circle = 'perimeter'
        # if fit_circle == 'area':
        #     # 1.5 because the circle looks small
        #     r = math.sqrt(1.2 * area / math.pi)
        # elif fit_circle == 'perimeter':
        #     r = cv2.arcLength(c, True) / (2 * math.pi)
        circles = cv2.HoughCircles(mask,
                                   cv2.HOUGH_GRADIENT,
                                   1,
                                   20,
                                   param1=param1,
                                   param2=param2)
        if circles is not None:
            # print('Circles:', circles)
            x, y, r = circles[0][0]
        else:
            x = y = r = 0
            self.exists = False
        return x, y, r

    @property
    def ipos(self):
        return (int(self.x), int(self.y))

    @classmethod
    def reset_id(cls):
        cls.id_cnt = 0

    def check_exists(self, frame, frame_idx):
        circle = np.zeros(frame.shape, dtype=np.uint8)
        circle = cv2.circle(circle, (int(self.x), int(self.y)), int(self.r),
                            (255, 255, 255), -1)
        circle = circle[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        cv2.erode(circle, kernel=None, iterations=5)
        frame = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        result = cv2.bitwise_and(frame, circle)
        self.exists = np.any(result)
        if not self.exists:
            self.lifetime = frame_idx - self.init_idx

    def kill(self, frame_idx):
        self.exists = False
        self.lifetime = frame_idx - self.init_idx


class AnalyzeBubblesLifetime(Analysis):

    def __init__(self, **opts):
        opts['name'] = 'LifetimeAnalysis'
        self.save_dir = "lifetime_export2"

        self.dir = "source\\Large Trimmed"
        self.file_idx = 16
        self.video_idx_param = ptypes.SimpleParameter(type='int',
                                                      name='video_idx',
                                                      value=self.file_idx,
                                                      readonly=True)
        self.video_list = [
            f for f in os.listdir(self.dir) if f.endswith('.mp4')
        ]
        self.video_list = sorted(self.video_list,
                                 key=lambda x: int(x.split('-')[0]))
        self.num_videos = len(self.video_list)
        print(self.video_list)
        opts['url'] = f"{self.dir}\\{self.video_list[self.file_idx]}"

        self.filters = [
            fp.Normalize(),
            fp.Blur(radius=2),
            fp.Threshold(type='thresh', lower=83),
            fp.Invert(),
            # fp.Dilate(),
            # fp.Erode()
        ]

        # analysis params
        self.view_debug = ptypes.SimpleParameter(type='bool',
                                                 name='debug_view',
                                                 title='Debug View',
                                                 value=True)
        self.state_opts = [
            'Init', 'Difference', 'Bubble Start', 'Threshold', 'Segmentation',
            'Export', 'Lifetime', 'Done'
        ]
        self.analysis_state_param = ptypes.ListParameter(
            name='analysis_state',
            title='Analysis State',
            value='Init',
            limits=self.state_opts)

        self.hough_param1 = ptypes.SimpleParameter(type='int',
                                                   name='HoughParam1',
                                                   value=20)
        self.hough_param2 = ptypes.SimpleParameter(type='int',
                                                   name='HoughParam2',
                                                   value=12)
        self.canny_param1 = ptypes.SimpleParameter(type='int',
                                                   name='CannyParam1',
                                                   value=20)
        self.canny_param2 = ptypes.SimpleParameter(type='int',
                                                   name='CannyParam2',
                                                   value=20)

        self.median_blur_param = ptypes.SimpleParameter(type='int',
                                                        name='MedianBlur',
                                                        value=3)

        self.debug_preview_param = ptypes.ListParameter(name='view_list',
                                                        value='',
                                                        limits=[])

        # overlay params
        self.toggle_center_param = ptypes.SimpleParameter(type='bool',
                                                          name='Toggle Center',
                                                          value=True)
        self.toggle_edge_param = ptypes.SimpleParameter(type='bool',
                                                        name='Toggle Edge',
                                                        value=True)

        params = [{
            'title': 'Preprocessing',
            'name': 'filter_group',
            'type': 'FilterGroup',
            'expanded': False,
            'children': self.filters
        }, {
            'title':
            'Analysis Params',
            'name':
            'analysis',
            'type':
            'group',
            'children': [
                self.video_idx_param,
                {
                    'title': 'Next Analysis',
                    'name': 'next_analysis',
                    'type': 'action',
                },
                self.analysis_state_param,
                {
                    'title': 'Use DistTransform',
                    'name': 'use_dist',
                    'type': 'bool',
                    'value': True
                },
                {
                    'title': 'Peak Region Size',
                    'name': 'peak_footprint',
                    'type': 'slider',
                    'value': 15,
                    'limits': (1, 50),
                },
                {
                    'title': 'Shrink Boundary',
                    'name': 'shrink',
                    'type': 'slider',
                    'value': 2,
                    'limits': (0, 10),
                },
                self.hough_param1,
                self.hough_param2,
                self.median_blur_param,
                self.canny_param1,
                self.canny_param2,
                self.debug_preview_param,
            ]
        }, {
            'title':
            'Overlay',
            'name':
            'overlay',
            'type':
            'group',
            'children': [{
                'title': 'Toggle',
                'name': 'toggle_overlay',
                'type': 'bool',
                'value': True
            }, {
                'title': 'Full Scale',
                'name': 'fs_view',
                'type': 'bool',
                'value': True
            }, {
                'title': 'Colormap',
                'name': 'colormap',
                'type': 'bool',
                'value': False
            }, {
                'title': 'Show ID',
                'name': 'show_id',
                'type': 'bool',
                'value': False
            }, self.view_debug, self.toggle_center_param,
                         self.toggle_edge_param]
        }]

        # check if children exists in case loading from saved state
        if 'children' not in opts:
            opts['children'] = params

        super().__init__(**opts)

        self.views = {}
        self.view_set = False
        self.export_cnt = 0
        self.saved_imgs_loc = ""

        self.bubbles = []

    def analyze(self, frame):
        frame = frame.cvt_color('gray')
        super().analyze(frame)  # sets orig.frame

        # take the difference
        if self.analysis_state_param.value() == 'Difference':
            self.child('filter_group', 'Normalize').set_normalized()
            self.views['filtered'] = self.crop_to_roi(
                self.child('filter_group').preprocess(frame))
        # set frame index to start of bubble analysis
        elif self.analysis_state_param.value() == 'Bubble Start':
            self.views['filtered'] = self.crop_to_roi(
                self.child('filter_group').preprocess(frame))
            self.child('filter_group', 'view_list').setValue('Blur')
            self.get_url_info(self.file_param.value())
            self.curr_frame_idx = self.analysis_start_idx
        # segment initial bubble positions and regions
        elif self.analysis_state_param.value() == 'Threshold':
            print("Thresholding auto")
            print(
                self.child('filter_group', 'Threshold', 'thresh_type').value())
            self.views['filtered'] = self.crop_to_roi(
                self.child('filter_group').preprocess(frame))
            self.view_set = False
        elif self.analysis_state_param.value() == 'Segmentation':
            frame = self.child('filter_group').preprocess(frame)

            nonzero = np.count_nonzero(frame)
            total = frame.size
            ratio = nonzero / total
            print(ratio)

            if ratio > 0.3:
                print('Threshold value too high, try again')
                self.views['filtered'] = self.crop_to_roi(frame)
                self.view_set = False
                return

            # self.set_auto_roi(frame)
            # cv2.imshow('frame', frame)
            self.views['filtered'] = self.crop_to_roi(frame)

            if self.child('analysis', 'use_dist').value():
                self.views['dist'] = ndi.distance_transform_edt(
                    self.views['filtered'])
            else:
                blur = self.crop_to_roi(
                    self.child('filter_group').get_preview('Blur'))
                blur = self.fs_stretch(blur)
                blur = 255 - blur  # invert
                blur[self.views['filtered'] == 0] = 0
                self.views['dist'] = blur
            fp = self.child('analysis', 'peak_footprint').value()
            # take the peak brightness in the distance transforms as that would be around the center of the bubble
            coords = peak_local_max(self.views['dist'],
                                    footprint=np.ones((fp, fp)),
                                    labels=self.views['filtered'])
            self.views['seed'] = np.zeros(self.views['dist'].shape,
                                          dtype=np.uint8)
            self.views['seed'][tuple(coords.T)] = True
            # for seeds that are too close to each other, merge them
            self.views['seed'] = cv2.dilate(self.views['seed'],
                                            kernel=None,
                                            iterations=1)
            markers, _ = ndi.label(self.views['seed'])
            self.views['watershed'] = watershed(-self.views['dist'],
                                                markers,
                                                mask=self.views['filtered'])
            self.bubbles = []
            Bubble.reset_id()
            self.max_bubbles = 120
            # ignore 0: background, get all other labels
            for label in np.unique(
                    self.views['watershed'])[1:self.max_bubbles]:
                mask = np.zeros(self.views['watershed'].shape, dtype='uint8')
                mask[self.views['watershed'] == label] = 255
                # shrink = cv2.erode(mask, kernel=None, iterations=1)
                # edge = cv2.bitwise_xor(shrink, mask)
                self.bubbles.append(
                    Bubble(mask, self.curr_frame_idx,
                           self.child('analysis', 'shrink').value(),
                           self.median_blur_param.value(),
                           self.hough_param1.value(),
                           self.hough_param2.value()))
            # blur = self.crop_to_roi(
            #     self.child('filter_group').get_preview('Blur'))
            # self.views['edge'] = cv2.Canny(blur, self.canny_param1.value(),
            #                                self.canny_param2.value())
            # circles = cv2.HoughCircles(self.views['edge'],
            #                            cv2.HOUGH_GRADIENT,
            #                            1,
            #                            20,
            #                            param1=self.hough_param1.value(),
            #                            param2=self.hough_param2.value())
            # if circles is not None:
            #     circles = np.uint16(np.around(circles[0]))
            #     print(circles)
            #     circle_frame = np.zeros(frame.shape, dtype=np.uint8)
            #     for c in circles:
            #         circle_frame = cv2.circle(circle_frame,
            #                                   (int(c[0]), int(c[1])),
            #                                   int(c[2]), 255, 2)
            #     self.views['circle'] = circle_frame
            self.view_set = False
        # 8778912686: hillenbrand number for
        # set overlays for exporting reference frame
        elif self.analysis_state_param.value() == 'Export':
            pass
            # self.view_debug.setValue(False)
            # self.child('filter_group', 'view_list').setValue('Blur')
            # self.child('overlay', 'show_id').setValue(True)
        # analyze lifetime
        elif self.analysis_state_param.value() == 'Lifetime':
            self.views['filtered'] = self.crop_to_roi(
                self.child('filter_group').preprocess(frame))
            empty = True
            for b in self.bubbles:
                # if existed in previous frame
                if b.exists:
                    if self.has_ended:
                        b.kill(self.curr_frame_idx)
                    else:
                        # check if current frame it's still there
                        empty = False
                        b.check_exists(self.views['filtered'],
                                       self.curr_frame_idx)

            if empty:
                self.analysis_state_param.setValue('Done')
                self.export_data()
                self.is_playing = False
        # on the first run through display all debug frames
        if not self.view_set:
            keys = list(self.views.keys())
            self.debug_preview_param.setLimits(keys)
            # if len(keys):
            #     self.debug_preview_param.setValue(keys[0])
            self.view_set = True

    def annotate(self, frame):
        curr_analysis_state = self.analysis_state_param.value()
        # print('Analysis:', curr_analysis_state)
        # cv2.imshow(curr_analysis_state, frame)

        # analysis state edge triggered actions.
        # these state changes happen automatically after analysis is done
        if curr_analysis_state == 'Init':
            self.video_idx_param.setValue(self.file_idx)
            self.curr_frame_idx = 0
            self.analysis_state_param.setValue('Difference')
            self.request_analysis_update.emit()
            return frame
        elif curr_analysis_state == 'Difference':
            self.analysis_state_param.setValue('Bubble Start')
            self.request_analysis_update.emit()
            return frame
        elif curr_analysis_state == 'Bubble Start':
            self.child('filter_group', 'Blur', 'radius').setValue(3)
            # self.child('filter_group', 'Threshold',
            #            'thresh_type').setValue('otsu')
            self.analysis_state_param.setValue('Threshold')
            # cv2.waitKey(10)
            self.request_analysis_update.emit()
            return frame
        elif curr_analysis_state == 'Threshold':
            self.child('filter_group', 'Threshold',
                       'thresh_type').setValue('thresh')
            self.analysis_state_param.setValue('Segmentation')
            self.request_analysis_update.emit()
            return frame
        # getting ready for reference frame export
        # don't return frame yet, wait until end
        elif curr_analysis_state == 'Export':
            self.analysis_state_param.setValue('Lifetime')
            self.request_analysis_update.emit()
            frame = self.crop_to_roi(
                self.child('filter_group').get_preview('Blur'))
            frame = self.fs_stretch(frame).cvt_color('bgr')
            frame_with_id = frame.copy()

            for b in self.bubbles:
                shrink = cv2.erode(b.init_mask, kernel=None, iterations=1)
                edge = cv2.bitwise_xor(shrink, b.init_mask)
                # print(shrink.shape, edge.shape)
                if b.exists:
                    edge_color = (0, 255, 0)
                else:
                    edge_color = (255, 0, 0)
                frame[edge == 255] = edge_color
                cv2.circle(frame, b.ipos, 1, (255, 255, 0), 1)
            for b in self.bubbles:
                cv2.circle(frame_with_id, b.ipos, int(b.r), (255, 0, 0), 1)
                cv2.putText(frame_with_id, str(b.id),
                            (int(b.x) - 11, int(b.y) + 7),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            self.export_frame(frame.cvt_color('rgb'),
                              frame_with_id.cvt_color('rgb'))
            return frame
        # analysis has finished
        elif curr_analysis_state == 'Done':
            if self.file_idx <= self.num_videos - 1:
                print("next video")
                self.file_idx += 1
                self.video_idx_param.setValue(self.file_idx)
                url = f"{self.dir}\\{self.video_list[self.file_idx]}"
                self.file_param.setValue(url)
            else:
                print('all videos analyzed')

        if self.view_debug.value() and self.views:
            # print('View key:', self.debug_preview_param.value(),
            #       self.views.keys())
            view_frame = self.views[self.debug_preview_param.value()]
        else:
            view_frame = self.crop_to_roi(
                self.child('filter_group').get_preview())

        x, y = self.cursor_pos
        if (0 <= y < view_frame.shape[0] and 0 <= x < view_frame.shape[1]):
            self.set_cursor_value(self.cursor_pos, view_frame[y, x])

        if view_frame is not None:
            # don't crop because the images stored in the analysis are already cropped
            frame = MyFrame(view_frame).cvt_color('bgr')
        else:
            frame = self.crop_to_roi(frame.cvt_color('bgr'))

        if self.child('overlay', 'fs_view').value():
            frame = self.fs_stretch(frame)

        if self.child('overlay', 'colormap').value():
            frame = cv2.applyColorMap(np.uint8(frame), cv2.COLORMAP_JET)

        if self.child('overlay', 'toggle_overlay').value():
            for b in self.bubbles:
                if self.toggle_edge_param.value():
                    shrink = cv2.erode(b.init_mask, kernel=None, iterations=1)
                    edge = cv2.bitwise_xor(shrink, b.init_mask)
                    # print(shrink.shape, edge.shape)
                    if b.exists:
                        edge_color = (0, 255, 0)
                    else:
                        edge_color = (255, 0, 0)

                    frame[edge == 255] = edge_color

                if self.child('overlay', 'show_id').value():
                    cv2.putText(frame, str(b.id),
                                (int(b.x) - 11, int(b.y) + 7),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                if self.toggle_center_param.value():
                    # if b.exists:
                    # cv2.circle(frame, b.ipos, 1, (255, 255, 0), 1)
                    # else:
                    cv2.circle(frame, b.ipos, int(b.r), (255, 0, 0), 1)

        # if curr_analysis_state == 'Export':
        #     print('exporting ref frame')
        #     self.export_frame(frame.cvt_color('rgb'))

        return frame

    def on_param_change(self, parameter, changes):
        # super().on_param_change(parameter, changes)
        param, change, data = changes[0]
        parent = param.parent().name()
        name = param.name()

        if name == 'File Select':
            filename = os.path.basename(data)
            self.file_idx = self.video_list.index(filename)
            self.video_idx_param.setValue(self.file_idx)
            self.analysis_state_param.setValue('Init')
            self.frame_idx_param.setValue(0)
        elif name == 'next_analysis':
            if self.analysis_state_param.value() == 'Segmentation':
                self.analysis_state_param.setValue('Export')
                self.child('filter_group', 'Threshold',
                           'thresh_type').setValue('thresh')
                set_thresh = self.child('filter_group', 'Threshold',
                                        'upper').value() + 2
                print(set_thresh)
                self.child('filter_group', 'Threshold',
                           'upper').setValue(set_thresh)
                self.child('filter_group', 'Blur', 'radius').setValue(2)
                self.curr_frame_idx += 1
                self.is_playing = True
            if (self.analysis_state_param.value() == 'Done'
                    and self.file_idx < self.num_videos):
                self.file_idx += 1
                url = f"{self.dir}\\{self.video_list[self.file_idx]}"
                self.file_param.setValue(url)

        if (parent == 'overlay' or name == 'debug_view'
                or (parent == 'analysis' and name == 'view_list')
                or name == 'cursor_info'
                or (parent == 'settings' and name == 'view_list')):
            self.request_annotate_update.emit()
        else:
            self.request_analysis_update.emit()

    def fs_stretch(self, frame):
        diff = frame.max() - frame.min()
        p = (255 + 1) / (diff + 1)  # + 1 to avoid div by 0
        a = frame.min() * p
        # print(f'p:{p}, a:{a}, min:{frame.min()}, max:{frame.max()}')
        return p * frame - a

    def video_ended(self, state):
        super().video_ended(state)
        if state:

            self.request_annotate_update.emit()

    def set_auto_roi(self, frame):
        coords = np.nonzero(frame)
        if len(coords[0]):
            (h, w) = frame.shape[:2]
            top_y = min(np.max(coords[0]) + 10, h)
            bot_y = max(np.min(coords[0]) - 10, 0)
            self.opts['roi'] = [0, bot_y, w, top_y - bot_y]

    def get_url_info(self, url):
        filename = os.path.basename(url).split('.')[0]
        print(filename)
        self.exposure_time, val1, val2 = filename.split('-')
        self.video_start_idx = int(val1)
        self.video_bubble_idx = int(val2)
        self.analysis_start_idx = self.video_bubble_idx - self.video_start_idx

    def export_data(self):
        self.fps_orig_video = 169
        self.um_per_px = 290.25 / 1280
        self.ms_per_frame = 1000 / self.fps_orig_video
        self.dist_units = 'um'
        self.time_units = 'ms'

        if self.dist_units == 'um':
            conversion = self.um_per_px
        elif self.dist_units == 'px':
            conversion = 1

        if self.time_units == 'ms':
            time_conversion = self.ms_per_frame
        else:
            time_conversion = 1

        if self.export_cnt > 0:
            mode = 'a'
        else:
            mode = 'w'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        with pd.ExcelWriter(f'{self.save_dir}/bubble_lifetime.xlsx',
                            engine="openpyxl",
                            mode=mode) as w:
            df = pd.DataFrame()
            df['id'] = [b.id for b in self.bubbles]
            df[f'x ({self.dist_units})'] = [
                b.x * conversion for b in self.bubbles
            ]
            df[f'y ({self.dist_units})'] = [
                b.y * conversion for b in self.bubbles
            ]
            df[f'r ({self.dist_units})'] = [
                b.r * conversion for b in self.bubbles
            ]
            df[f'lifetime ({self.time_units})'] = [
                b.lifetime * time_conversion for b in self.bubbles
            ]
            df = df.sort_values(by=[f'x ({self.dist_units})'])
            diff = df.diff()
            df['x dist'] = diff[f'x ({self.dist_units})']
            df['y dist'] = diff[f'y ({self.dist_units})']
            df['euclid'] = (df['x dist'].mul(df['x dist']).add(
                df['y dist'].mul(df['y dist'])))**(1 / 2)
            print(df)
            sheet_name = f'{self.exposure_time}ms-{self.video_bubble_idx}f'
            df.to_excel(w, sheet_name=sheet_name, index=False)
            print('exported')
            workbook = w.book
            ws = workbook[sheet_name]
            print(self.saved_imgs_loc)
            for i, path in enumerate(self.saved_imgs_loc):
                img = openpyxl.drawing.image.Image(path)
                img.anchor = f'J{10*i+3}'
                ws.add_image(img)

        self.export_cnt += 1

    def export_frame(self, *frames):
        folder = f"{self.save_dir}\\reference"
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.saved_imgs_loc = []
        for i, frame in enumerate(frames):
            name = f'{folder}\\{self.exposure_time}ms-{self.video_bubble_idx}f-{i}.jpg'
            self.saved_imgs_loc.append(name)
            cv2.imwrite(name, frame)
