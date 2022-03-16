from multiprocessing import Event
from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QImage
from pyqtgraph.parametertree.parameterTypes import TextParameterItem
import cv2
import os
from queue import Queue
import time
from analysis_params import Analysis

from misc_methods import MyFrame
# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
# - jk just make sure thread doesn't do things too quickly
# - by that I mean put a 1ms delay in infinite while loops


class ImageProcessingThread(QThread):
    view_frame = pyqtSignal(str, object)
    url_updated = pyqtSignal()
    on_new_frame = pyqtSignal(int)  # curr frame index

    video_extensions = ['.mp4', '.avi']
    img_extensions = ['.jpg', '.jpeg', '.png']

    def __init__(self, parent, url, analysis: Analysis):
        super().__init__(parent)
        self.main_param = analysis
        self.orig_frame = None
        self.cropped_orig = None
        self.frame_to_show = None
        self.cursor_pos = [0, 0]
        self.scaling = 1

        # video properties
        self.prev_frame_time = 0
        self.frame_interval = 0  # period between frames based on fps
        self.is_paused = True
        self.update_once_flag = False
        self.start_frame_idx = 0
        self.end_frame_idx = 10e9

        # image properties
        self.new_image_flag = False

        # thread processing properties
        self.annotate_frame_flag = False
        self.exit_flag = False
        self.analyze_frame_flag = True
        self.export_frame_flag = False
        self.frame_is_processed = False

        self.main_param.request_analysis_update.connect(
            lambda: self.set_update_flag(analyze=True, annotate=True))
        self.main_param.request_annotate_update.connect(
            lambda: self.set_update_flag(analyze=False, annotate=True))
        self.main_param.request_set_frame_idx.connect(
            self.set_curr_frame_index)

        self.window_name = 'Preview'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_url(url)

    # in image mode, perform analysis everytime parameters are changed
    # in video mode, perform analysis everytime frame updates
    #   - make a request to main for the operations based on params every frame
    def update_url(self, url):
        self.source_url = url
        self.split_url = os.path.splitext(url)
        # is video
        if self.split_url[1] in self.video_extensions:
            self.video_cap = cv2.VideoCapture(url)
            self.vid_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.frame_interval = 1 / self.vid_fps
            self.update_once_flag = True
            self.end_frame_idx = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print('Video Mode, fps:', self.vid_fps)
            print('End frame:', self.end_frame_idx)
        # is image
        elif self.split_url[1] in self.img_extensions:
            self.new_image_flag = True
            self.video_cap = None

        print(self.split_url)

    def set_curr_frame_index(self, index):
        print('orig', index)
        idx = int(min(index, self.end_frame_idx - 1))
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            print('setting frame to', idx)
            self.update_once_flag = True

    def mouse_callback(self, event, x, y, flags, param):
        self.cursor_pos = [int(x / self.scaling), int(y / self.scaling)]
        if event == cv2.EVENT_MOUSEMOVE:
            self.main_param.on_mouse_move_event(*self.cursor_pos)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.main_param.on_mouse_click_event('left')
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.main_param.on_mouse_click_event('right')

    def set_playback_state(self, resume):
        self.is_paused = not resume

    def draw_cursor(self, frame):
        return cv2.circle(frame, self.cursor_pos, 1, (255, 255, 255), -1)

    def update_view(self, frame):
        self.width = 900
        (h, w, _) = frame.shape
        self.scaling = self.width / w
        dim = (self.width, int(h * self.scaling))
        frame = self.draw_cursor(frame)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        self.view_frame.emit(self.window_name, resized)

    def set_update_flag(self, analyze, annotate):
        self.analyze_frame_flag = analyze
        self.annotate_frame_flag = annotate

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
                if ((self.update_once_flag or self.main_param.is_playing())
                        and time_elapsed >= self.frame_interval):
                    # Playing: takes in synchronous updates:
                    # only updates view when the time has reached
                    # the frame interval period
                    self.prev_frame_time = time.time()
                    self.update_once_flag = False
                    self.main_param.curr_frame_idx = self.video_cap.get(
                        cv2.CAP_PROP_POS_FRAMES)
                    ret, frame = self.video_cap.read()
                    if ret:
                        self.orig_frame = MyFrame(frame, 'bgr')
                        self.analyze_frame_flag = True
            # there is an image object
            else:
                if self.new_image_flag:
                    self.orig_frame = MyFrame(cv2.imread(self.source_url),
                                              'bgr')
                    self.new_image_flag = False

            # simply analyze the current frame in memory
            if self.analyze_frame_flag:
                self.main_param.analyze(
                    self.orig_frame.copy())
                self.analyze_frame_flag = False
                # if analyzed we will want to update view
                self.annotate_frame_flag = True

            # draws on any new annotations and then sends it to
            # main window to display
            if self.annotate_frame_flag:
                show = self.main_param.annotate(self.orig_frame.copy())
                self.update_view(show)
                self.annotate_frame_flag = False

        cv2.destroyAllWindows()

    def stop(self):
        self.exit_flag = True
        if self.video_cap is not None:
            self.video_cap.release()
        self.wait()
