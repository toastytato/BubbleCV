from cmath import inf
import sys
from tkinter import Frame
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QGridLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QApplication,
    QMainWindow,
    QPushButton,
    QLineEdit,
)
from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
from pyqtgraph.parametertree.parameterTypes import TextParameterItem
import cv2
import os
from queue import Queue
import sys
import time

from matplotlib.pyplot import hlines

# --- my classes ---
from main_params import RESET_DEFAULT_PARAMS, MyParams
from filters import *
from bubble_analysis import get_save_dir
from misc_methods import MyFrame
from analysis_params import Analysis, AnalyzeBubbleLaser, AnalyzeBubblesWatershed

# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
# - jk just make sure thread doesn't do things too quickly
# - by that I mean put a 1ms delay in infinite while loops


class ImageProcessingThread(QThread):
    changePixmap = pyqtSignal(QImage)
    url_updated = pyqtSignal()
    roi_updated = pyqtSignal(object)
    update_q_request = pyqtSignal()
    on_new_frame = pyqtSignal(int)  # curr frame index

    video_extensions = ['.mp4', '.avi']
    img_extensions = ['.jpg', '.jpeg', '.png']

    def __init__(self, parent, url, weight=0, roi=None):
        super().__init__(parent)
        self.analysis_q = Queue(10)  # max operatio
        self.annotate_q = Queue(10)
        self.orig_frame = None
        self.cropped_orig = None
        self.frame_to_show = None
        self.roi = roi
        self.weight = weight
        self.cursor_x = 0
        self.cursor_y = 0

        # video properties
        self.prev_frame_time = 0
        self.frame_interval = 0  # period between frames based on fps
        self.is_paused = False
        self.update_once_flag = False
        self.curr_frame_idx = 0
        self.start_frame_idx = 0
        self.end_frame_idx = inf

        # image properties
        self.new_image_flag = False

        # thread processing properties
        self.show_frame_flag = False
        self.exit_flag = False
        self.analyze_frame_flag = False
        self.export_frame_flag = False
        self.select_roi_flag = False
        self.frame_is_processed = False
        self.update_url(url)

    def start_image_operations(self):
        # to prevent adding to queue before it can be processed
        if self.isRunning():
            self.analyze_frame_flag = True

    def add_to_analysis_queue(self, op):
        if not self.analyze_frame_flag:
            self.analysis_q.put(op)

    def add_to_annotate_queue(self, op):
        if not self.show_frame_flag:
            self.annotate_q.put(op)

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
        self.curr_frame_idx = int(min(index, self.end_frame_idx-1))
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame_idx)
            self.update_once_flag = True

    def get_roi(self):
        self.select_roi_flag = True
        if self.video_cap:
            self.update_once_flag = True

    # tells thread to export instead of main loop
    def export_curr_frame(self):
        self.export_frame_flag = True

    @pyqtSlot(bool)
    def set_pause_state(self, state):
        self.is_paused = state

    @pyqtSlot(int, int)
    def update_cursor(self, x, y):
        self.cursor_x = x
        self.cursor_y = y

    def set_overlay_weight(self, weight):
        self.weight = weight
        self.show_frame_flag = True

    def _draw_cursor(self, frame):
        return MyFrame(
            cv2.circle(frame, (self.cursor_x, self.cursor_y), 1,
                       (255, 255, 255), -1))

    def _export_frame(self, frame=None):
        print('export frame')
        path = get_save_dir('analysis', self.source_url) + "/overlay.png"
        if frame is None:
            export = self.cropped_orig.copy()
            # export[self.annotated > 0] = self.annotated[self.annotated > 0]
            cv2.imwrite(path, export)
        else:
            cv2.imwrite(path, frame)

    # CNN segmentation yet to be implemented
    def export_frame_for_training(self, **kwargs):
        file_name = os.path.basename(self.split_url[0])
        mask_path = get_save_dir('training', 'mask') + f'/{file_name}_mask.png'
        orig_path = get_save_dir('training', 'frame') + f'/{file_name}.png',
        # if no frame passed in param, use object var
        # cv2.imwrite(mask_path, kwargs.get('orig', self.cropped_orig))
        # cv2.imwrite(orig_path, kwargs.get('mask', self.annotated))

    # method runs in new thread, called on thread.start()
    def run(self):
        while not self.exit_flag:

            # I think this logic could be optimized ehheheh
            # processing video
            if self.video_cap:
                # meet the original video framerate
                if ((time.time() - self.prev_frame_time >= self.frame_interval
                     and not self.is_paused and self.frame_is_processed)
                    or self.update_once_flag):
                    # when interval is ready, read frame if not paused
                    # or update once when new file selected or manual frame changes
                    self.prev_frame_time = time.time()
                    self.update_once_flag = False
                    ret, self.orig_frame = self.video_cap.read()
                    self.orig_frame = MyFrame(self.orig_frame, 'bgr')
                    self.on_new_frame.emit(self.curr_frame_idx)
                    self.curr_frame_idx += 1
                    # valid frame
                    if ret:
                        self.update_q_request.emit()
                        self.frame_is_processed = False
                    # reached end of video (will ret false)
                    elif self.curr_frame_idx >= self.video_cap.get(
                            cv2.CAP_PROP_FRAME_COUNT):
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES,
                                           self.start_frame_idx)
                        self.curr_frame_idx = self.start_frame_idx
                        print("End of video reached, restarting")
                        continue
                    # frame is invalid while in middle of clip
                    else:
                        cv2.waitKey(
                            1)  # waiting 1 ms prevents UI from locking up
                        continue  # ignore processing
                elif self.is_paused or not self.frame_is_processed:
                    cv2.waitKey(1)
                    # continue processing frame when paused
                else:
                    cv2.waitKey(1)  # waiting 1 ms prevents UI from locking up
                    continue  # ignore processing
            # processing image
            else:
                # prevents the thread from locking up UI when in img mode
                cv2.waitKey(1)  # waiting 1 ms speeds up UI side
                # when new image is selected
                if self.new_image_flag:
                    self.orig_frame = MyFrame(cv2.imread(self.source_url),
                                              'bgr')
                    self.new_image_flag = False

            # if self.export_frame_flag:
            #     export = self.frame_to_show.copy()
            #     self._export_frame(export)
            #     self.export_frame_for_training(orig=self.cropped_orig,
            #                                    mask=self.frame_to_show)
            #     self.export_frame_flag = False

            # selectROI blocks loop while waiting for user to select
            if self.select_roi_flag and self.orig_frame is not None:
                r = cv2.selectROI("Select ROI", self.orig_frame)
                if all(r) != 0:
                    self.roi = r
                cv2.destroyWindow("Select ROI")
                self.roi_updated.emit(self.roi)
                self.select_roi_flag = False
                self.analyze_frame_flag = True

            # get cropped frame whenever displaying frame
            if self.analyze_frame_flag or self.show_frame_flag:
                # get cropped frame
                if len(self.roi) > 0:
                    self.cropped_orig = self.orig_frame[
                        int(self.roi[1]):int(self.roi[1] + self.roi[3]),
                        int(self.roi[0]):int(self.roi[0] + self.roi[2])]
                else:
                    self.cropped_orig = self.orig_frame

                # start processing frame
                if self.analyze_frame_flag:
                    while not self.analysis_q.empty():
                        p = self.analysis_q.get()
                        p(self.cropped_orig, self.curr_frame_idx)
                    self.analyze_frame_flag = False
                    self.show_frame_flag = True

                if self.show_frame_flag:
                    self.frame_to_show = self.cropped_orig.copy()
                    while not self.annotate_q.empty():
                        p = self.annotate_q.get()
                        self.frame_to_show = p(self.frame_to_show)

                    self.show_frame_flag = False
                    self.frame_is_processed = True

                    show = self._draw_cursor(self.frame_to_show)
                    # cv2.imshow("Frame", frame)
                    rgbImage = show.cvt_color('rgb')
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    qt_img = QImage(rgbImage.data, w, h, bytesPerLine,
                                    QImage.Format_RGB888)
                    self.changePixmap.emit(qt_img)

        cv2.destroyAllWindows()

    def stop(self):
        self.exit_flag = True
        if self.video_cap:
            self.video_cap.release()
        self.wait()


# =---------------------------------------------------------
class DisplayTimelineWidget(QWidget):
    pause_signal = pyqtSignal(bool)
    set_frame_signal = pyqtSignal(int)

    IS_PAUSED_TEXT = "Play"
    IS_PLAYING_TEXT = "Pause"
    FWD_TEXT = ">>"
    REV_TEXT = "<<"

    def __init__(self, parent):
        super().__init__(parent)

        self.is_playing = False

        h_layout = QHBoxLayout()

        self.prev_frame_btn = QPushButton(self.REV_TEXT)
        self.play_pause_btn = QPushButton()
        # highlight btn when playing
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.setChecked(self.is_playing)
        self.next_frame_btn = QPushButton(self.FWD_TEXT)

        self.play_pause_btn.clicked.connect(self.on_play_pause_pressed)

        h_layout.addStretch()
        # h_layout.addWidget(self.prev_frame_btn)
        h_layout.addWidget(self.play_pause_btn)
        # h_layout.addWidget(self.next_frame_btn)
        h_layout.addStretch()

        self.setLayout(h_layout)
        self.setFixedHeight(50)

    def on_play_pause_pressed(self, event):
        print(event)
        self.is_playing = event
        if self.is_playing:
            self.play_pause_btn.setText(self.IS_PLAYING_TEXT)
        else:
            self.play_pause_btn.setText(self.IS_PAUSED_TEXT)

        # emit if it is paused
        self.pause_signal.emit(not self.is_playing)


class DisplayFrame(QLabel):
    mouse_moved_signal = pyqtSignal(int, int)
    mouse_pressed_signal = pyqtSignal(object)
    update_request_signal = pyqtSignal()
    image_set_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.width = 900
        self.height = 700
        self.resize(self.width, self.height)
        self.setMouseTracking(True)

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.img = image
        self.show = image.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.setPixmap(QPixmap.fromImage(self.show))
        self.resize(self.show.width(), self.height)
        self.image_set_signal.emit()

    def mouseMoveEvent(self, event):
        scale = self.img.width() / self.show.width()
        x = event.x() * scale
        y = event.y() * scale
        self.mouse_moved_signal.emit(int(x), int(y))
        self.update_request_signal.emit()

    def mousePressEvent(self, event):
        self.mouse_pressed_signal.emit(event)
        self.update_request_signal.emit()


# main ui, handles controller logic
class BubbleAnalyzerWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bubble Analyzer")

        default_url = "source\\022622_bubbleformation_80x_075M_3_original.avi"
        self.parameters = MyParams(default_url)
        self.parameters.paramChange.connect(self.on_param_change)
        self.init_ui()
        # process images separate from ui
        self.cv_thread = ImageProcessingThread(
            self,
            url=self.parameters.get_param_value("Settings", "File Select"),
            weight=self.parameters.get_param_value("Settings",
                                                   "Overlay Weight"),
            roi=self.parameters.internal_params["ROI"])
        self.cv_thread.changePixmap.connect(self.display_label.set_image)
        self.cv_thread.roi_updated.connect(self.on_roi_updated)
        self.cv_thread.update_q_request.connect(self.update_thread)
        self.cv_thread.on_new_frame.connect(self.on_new_frame)
        self.display_label.mouse_moved_signal.connect(
            self.cv_thread.update_cursor)
        # self.display_label.update_request_signal.connect(self.update_thread)
        self.display_label.image_set_signal.connect(
            lambda: self.resize(self.minimumSizeHint()))

        # self.display_timeline.set_frame_signal.connect(self.cv_thread.set_curr_frame_index)
        self.display_timeline.pause_signal.connect(
            self.cv_thread.set_pause_state)
        self.display_timeline.play_pause_btn.clicked.emit(
            False)  # set to paused

        # connect mouse events to params
        self.connect_param_signals()

        self.cv_thread.start()
        self.update_thread()

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        # self.setFixedHeight(620)
        self.layout = QGridLayout(self)

        self.display_label = DisplayFrame(self)
        self.display_timeline = DisplayTimelineWidget(self)
        # self.display_timeline.setSizePolicy(Qt)
        self.layout.addWidget(self.display_label,
                              0,
                              0,
                              alignment=Qt.AlignCenter)
        self.layout.addWidget(self.display_timeline, 1, 0, 2, 1)

        # display_layout.addWidget(self.display_timeline)
        # self.plotter = CenterPlot()
        self.parameters.setFixedWidth(400)
        # align center to make sure mouse coordinates maps to what's shown
        # self.layout.addWidget(self.display_label, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.parameters, 0, 1, 2, 1, Qt.AlignCenter)

        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)
        # self.resize(self.minimumSizeHint())

    def connect_param_signals(self):
        for param in self.parameters.params.child('Analyze').children():
            if isinstance(param, Analysis):
                param.request_play_state.connect(
                    self.display_timeline.play_pause_btn.setChecked)
                param.request_display_update.connect(
                    self.update_thread_from_signal)
                self.display_label.mouse_moved_signal.connect(
                    param.on_mouse_move_event)
                self.display_label.mouse_pressed_signal.connect(
                    param.on_mouse_click_event)
                self.cv_thread.roi_updated.connect(param.on_roi_updated)

    # using pyqtSlots not really necessary
    # but since this gets called many times
    # pyqtSlot offers a *slight* speed / memory bump
    @pyqtSlot(int)
    def on_new_frame(self, idx):
        param = self.parameters.get_child('Settings', 'curr_frame_idx')
        # update value in parameter tree without recursively calling itself
        self.parameters.disconnect_changes()
        param.setValue(idx)
        self.parameters.connect_changes()

    def on_roi_updated(self, roi):
        print('roi updated')
        self.parameters.internal_params["ROI"] = roi
        self.update_thread()

    # update frame canvas on param changes
    @pyqtSlot(object, object)
    def on_param_change(self, parameter, changes):
        has_operation = False
        print('changed')
        for param, change, data in changes:
            path = self.parameters.params.childPath(param)
            # print('Changes:', changes)
            # print('Param:', parameter)
            if path is None:
                continue
            if change == 'childRemoved':
                # on remove data is the object
                if isinstance(data, Analysis):
                    data.request_display_update.disconnect(
                        self.update_thread_from_signal)
                    self.display_label.mouse_moved_signal.disconnect(
                        data.on_mouse_move_event)
                    self.display_label.mouse_pressed_signal.disconnect(
                        data.on_mouse_click_event)
                    self.cv_thread.roi_updated.disconnect(data.on_roi_updated)

                has_operation = True
                continue
            if change == 'childAdded':
                # on add data is a tuple containing object(s) added
                # "tuple" even though theres really only one object
                for p in data:
                    if isinstance(p, Analysis):
                        print('connecting')
                        p.request_display_update.connect(
                            self.update_thread_from_signal)
                        self.display_label.mouse_moved_signal.connect(
                            p.on_mouse_move_event)
                        self.display_label.mouse_pressed_signal.connect(
                            p.on_mouse_click_event)
                        self.cv_thread.roi_updated.connect(p.on_roi_updated)
                has_operation = True
                continue

            if path[0] == "Settings":
                if path[1] == "File Select":
                    self.cv_thread.update_url(data)
                    self.parameters.update_url(data)
                    has_operation = True
                elif path[1] == 'curr_frame_idx':
                    self.cv_thread.set_curr_frame_index(data)
                elif path[1] == "Overlay Weight":
                    self.cv_thread.set_overlay_weight(data)
                elif path[1] == "Select ROI":
                    self.cv_thread.get_roi()
            if path[0] == "Analyze":
                has_operation = True
                if path[1] == "Bubbles":
                    if path[2] == "Export Distances":
                        self.cv_thread.export_frame_flag = True
                if path[1] == 'BubbleWatershed':
                    if path[2] == 'export_csv' or path[2] == 'export_graphs':
                        self.cv_thread.export_frame_flag = True

        if has_operation:
            self.update_thread()

    # for some reason params connected to signals get overwritten by
    # the default params.
    def update_thread_from_signal(self, analyze, annotate):
        self.update_thread(analyze, annotate)

    @pyqtSlot()
    def update_thread(self, analyze=True, annotate=True):
        # print('updating', analyze, annotate)
        if analyze:
            self.update_analysis()
        if annotate:
            self.update_annotations()
        self.cv_thread.analyze_frame_flag = analyze
        self.cv_thread.show_frame_flag = annotate

    def update_analysis(self):
        # op is a custom param object which contains the method analyze
        for op in self.parameters.params.child("Analyze").children():
            self.cv_thread.add_to_analysis_queue(op.analyze)

    def update_annotations(self):
        # draw on annotations at the very end
        # op is a custom param object which contains the method annotate
        for op in self.parameters.params.child("Analyze").children():
            self.cv_thread.add_to_annotate_queue(op.annotate)

    def closeEvent(self, event):
        cv2.destroyAllWindows()
        self.cv_thread.stop()
        self.parameters.save_settings()


def main():
    app = QApplication(sys.argv)
    main_window = BubbleAnalyzerWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
