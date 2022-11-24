from multiprocessing import Event
from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QImage
from pyqtgraph.parametertree.parameterTypes import TextParameterItem
import pyqtgraph as pg
import cv2
import os
import numpy as np
from queue import Queue
import time
from analysis_params import Analysis

from misc_methods import MyFrame

# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
# - jk just make sure thread doesn't do things too quickly
# - by that I mean put a 1ms delay in infinite while loops


# allows me to override keypress/mouse signals
class myImageView(pg.ImageView):

    keypress_signal = pyqtSignal(object)
    mouse_move_signal = pyqtSignal(object)
    mouse_click_signal = pyqtSignal(object)

    def __init__(self,
                 parent=None,
                 name="ImageView",
                 view=None,
                 imageItem=None,
                 levelMode='mono',
                 *args):
        blank_img = np.zeros((1, 1, 3), np.uint8)
        imageItem = self.myImageItem(image=blank_img)
        imageItem.mysigMouseClicked.connect(
            lambda ev: self.mouse_click_signal.emit(ev))
        imageItem.mysigMouseMoved.connect(
            lambda ev: self.mouse_move_signal.emit(ev))
        super().__init__(parent, name, view, imageItem, levelMode, *args)

    def keyPressEvent(self, ev):
        self.keypress_signal.emit(ev)
        return super().keyPressEvent(ev)

    class myImageItem(pg.ImageItem):
        # custom signal, emitted when mouse is clicked
        mysigMouseClicked = pyqtSignal(object)
        mysigMouseMoved = pyqtSignal(object)

        def __init__(self, image=None, **kargs):
            super().__init__(image, **kargs)

        def hoverEvent(self, ev):
            self.mysigMouseMoved.emit(ev)
            return super().hoverEvent(ev)

        def mouseMoveEvent(self, ev):
            self.mysigMouseMoved.emit(ev)
            return super().mouseMoveEvent(ev)

        def mouseClickEvent(self, ev):
            self.mysigMouseClicked.emit(ev)
            return super().mouseClickEvent(ev)


class ImageProcessingThread(QThread):
    view_frame = pyqtSignal(str, object)
    url_updated = pyqtSignal()
    on_new_frame = pyqtSignal(int)  # curr frame index

    video_extensions = ['.mp4', '.avi']
    img_extensions = ['.jpg', '.jpeg', '.png']

    def __init__(self, analysis: Analysis, imageView: myImageView, width=None):
        super().__init__()
        self.analysis_param = analysis
        self.orig_frame = None
        self.cropped_orig = None
        self.frame_to_show = None
        self.cursor_pos = [0, 0]
        self.scaling = 1
        self.width = width

        # video properties
        self.prev_frame_time = 0
        self.frame_interval = 0  # period between frames based on fps
        self.is_paused = True
        self.update_once_flag = False
        self.start_frame_idx = 0
        self.end_frame_idx = 10e9

        # thread processing properties
        self.exit_flag = False
        self.analyze_frame_flag = True
        self.annotate_frame_flag = False

        self.analysis_param.request_analysis_update.connect(
            lambda: self.set_update_flag(analyze=True))
        self.analysis_param.request_annotate_update.connect(
            lambda: self.set_update_flag(annotate=True))
        self.analysis_param.request_set_frame_idx.connect(
            self.set_curr_frame_index)
        self.analysis_param.request_url_update.connect(self.update_url)
        self.analysis_param.request_resume.connect(self.set_playback_state)

        self.img_view = imageView
        # Note: these signals are kinda scuffed
        # I had to modify the library code in order for this work
        self.img_view.mouse_click_signal.connect(self.mouse_click_callback)
        self.img_view.mouse_move_signal.connect(self.mouse_move_callback)
        self.img_view.keypress_signal.connect(self.keypress_callback)

        self.window_name = 'Preview'

        # can't update view in secondary thread, do this via signal so
        # operation is  performed in main thread with signals
        self.view_frame.connect(self.display_view)
        # cv2.namedWindow(self.window_name)
        # cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_url(self.analysis_param.opts['url'])
        self.set_curr_frame_index(self.analysis_param.curr_frame_idx)

    # in image mode, perform analysis everytime parameters are changed
    # in video mode, perform analysis everytime frame updates
    #   - make a request to main for the operations based on params every frame
    def update_url(self, url):
        print(url)
        self.source_url = url
        self.split_url = os.path.splitext(url)
        # is video
        if self.split_url[1] in self.video_extensions:
            self.video_cap = cv2.VideoCapture(url)
            self.vid_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.analysis_param.video_ended(False)
            # if self.vid_fps > 24:
            #     self.vid_fps = 24
            self.frame_interval = 1 / self.vid_fps
            self.update_once_flag = True
            self.end_frame_idx = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # idx = 0
            # self.video_data = []

            # # load all frames into memory
            # while True:
            #     ret, frame = self.video_cap.read()
            #     if ret:
            #         idx += 1
            #         if idx % 10 == 0:
            #             print(idx)
            #         self.video_data.append(frame)
            #     else:
            #         print('Finished adding video to memory')
            #         break

            # self.video_data = np.array(self.video_data)
            # print(self.video_data.shape)

            # self.display_view('Name', self.video_data)

            if self.width is None:
                self.width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print('Video Mode, fps:', self.vid_fps)
            print('End frame:', self.end_frame_idx)
        # is image
        elif self.split_url[1] in self.img_extensions:
            self.orig_frame = MyFrame(cv2.imread(self.source_url), 'bgr')
            if self.width is None:
                self.width = self.orig_frame.shape[1]
            self.video_cap = None
            self.analyze_frame_flag = True

        print(self.split_url)

    def set_curr_frame_index(self, index):
        idx = int(min(index, self.end_frame_idx - 1))
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            print('setting frame to', idx)
            self.update_once_flag = True

    def keypress_callback(self, event):
        print('keypress:', event.key())
        self.analysis_param.on_keypress_event(event.key())

    def mouse_move_callback(self, event):
        # print('move', event)
        try:
            pos = event.pos()
        except Exception as e:
            print('Mouse exception:', e)
            return
        self.cursor_pos = [int(pos[0]), int(pos[1])]
        self.analysis_param.on_mouse_move_event(*self.cursor_pos)

    def mouse_click_callback(self, event):
        if event.button() == 1:
            self.analysis_param.on_mouse_click_event('left')
        elif event.button() == 2:
            self.analysis_param.on_mouse_click_event('right')

    # # for opencv window mouse callbacks
    # def mouse_callback(self, event, x, y, flags, param):
    #     x -= 30
    #     self.cursor_pos = [int(x / self.scaling), int(y / self.scaling)]
    #     if event == cv2.EVENT_MOUSEMOVE:
    #         self.analysis_param.on_mouse_move_event(*self.cursor_pos)
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.analysis_param.on_mouse_click_event('left')
    #     elif event == cv2.EVENT_RBUTTONDOWN:
    #         self.analysis_param.on_mouse_click_event('right')

    def set_playback_state(self, resume):
        print('setting playback to:', resume)
        self.is_paused = not resume

    def draw_cursor(self, frame):
        return cv2.circle(frame, self.cursor_pos, 1, (255, 255, 255), -1)

    def display_view(self, name, frame):
        self.img_view.setImage(frame,
                               autoRange=False,
                               autoHistogramRange=False,
                               levels=(0, 255),
                               levelMode='mono',
                               axes={
                                   'y': 0,
                                   'x': 1,
                                   'c': 2
                               })

    def update_view(self, frame):
        # (h, w, _) = frame.shape
        # self.scaling = self.width / w
        # dim = (self.width, int(h * self.scaling))
        frame = MyFrame(frame)
        frame = self.draw_cursor(frame)
        # resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        self.view_frame.emit(self.window_name, frame)

    def set_update_flag(self, analyze=None, annotate=None):
        # print('update:', analyze, annotate)
        if analyze is not None:
            self.analyze_frame_flag = analyze
        if annotate is not None:
            self.annotate_frame_flag = annotate

    # performed in separate thread
    def run(self):
        while not self.exit_flag:
            cv2.waitKey(1)
            # there is a video object
            if self.video_cap is not None:
                # and playback is resumed
                # update once when new video is loaded
                # --> will update view even though paused

                # flipped because text will show pause when playing and
                # vice versa
                time_elapsed = time.time() - self.prev_frame_time
                if ((self.update_once_flag or self.analysis_param.is_playing)
                        and time_elapsed >= self.frame_interval):
                    # Playing: takes in synchronous updates:
                    # only updates view when the time has♠ reached
                    # the frame interval period
                    self.prev_frame_time = time.time()
                    self.update_once_flag = False
                    frame_idx = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self.analysis_param.set_curr_frame_idx_no_emit(frame_idx)
                    if frame_idx >= self.video_cap.get(
                            cv2.CAP_PROP_FRAME_COUNT):
                        print('Video Ended')
                        self.analysis_param.video_ended(True)
                    try:
                        ret, frame = self.video_cap.read()
                    except Exception as e:
                        print('Video Read Error:', e)
                    if ret:
                        self.orig_frame = MyFrame(frame, 'bgr')
                        self.analyze_frame_flag = True

            # simply analyze the current frame in memory
            if self.analyze_frame_flag:
                self.analysis_param.analyze(self.orig_frame.copy())
                self.analyze_frame_flag = False
                # if analyzed we will want to update view
                self.annotate_frame_flag = True

            # draws on any new annotations and then sends it to
            # main window to display
            if self.annotate_frame_flag:
                show = self.analysis_param.annotate(self.orig_frame.copy())
                self.update_view(show)
                self.annotate_frame_flag = False

        cv2.destroyAllWindows()

    def stop(self):
        self.exit_flag = True
        if self.video_cap is not None:
            self.video_cap.release()
        self.wait()
