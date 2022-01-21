import sys
from tkinter import Frame
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
from queue import Queue
import sys

# --- my classes ---
from main_params import RESET_DEFAULT_PARAMS, MyParams
from filters import *
from bubble_analysis import get_save_dir
from misc_methods import MyFrame
from analysis_params import AnalyzeBubblesWatershed

# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
# - jk just make sure thread doesn't do things too quickly
# - by that I mean put a 1ms delay in infinite while loops


class ImageProcessingThread(QThread):
    changePixmap = pyqtSignal(QImage)
    url_updated = pyqtSignal()
    roi_updated = pyqtSignal(object)

    def __init__(self, parent, url, weight=0, roi=None):
        super().__init__(parent)
        self.q = Queue(20)
        self.orig_frame = None
        self.filtered = None
        self.cropped_orig = None
        self.roi = roi
        self.weight = weight
        self.cursor_x = 0
        self.cursor_y = 0

        self.show_frame_flag = False
        self.exit_flag = False
        self.url_updated_flag = False
        self.start_processing_flag = False
        self.export_frame_flag = False
        self.select_roi_flag = False
        self.update_url(url)

    def start_image_operations(self):
        if self.isRunning():
            self.start_processing_flag = True

    def add_to_queue(self, op):
        if not self.start_processing_flag:
            self.q.put(op)

    def update_url(self, url):
        self.source_url = url
        self.split_url = os.path.splitext(url)
        self.url_updated_flag = True
        print(self.split_url)

    def get_roi(self):
        self.select_roi_flag = True

    def set_overlay_weight(self, weight):
        self.weight = weight
        self.show_frame_flag = True

    # allow for external calls without frame parameter
    def export_frame(self, frame=None):
        print('export frame')
        path = get_save_dir('analysis', self.source_url) + "/overlay.png"
        if frame is None:
            export = self.cropped_orig.copy()
            export[self.annotated > 0] = self.annotated[self.annotated > 0]
            cv2.imwrite(path, export)
        else:
            cv2.imwrite(path, frame)

    # CNN segmentation yet to be implemented
    def export_frame_for_training(self, **kwargs):
        file_name = os.path.basename(self.split_url[0])
        mask_path = get_save_dir('training', 'mask') + f'/{file_name}_mask.png'
        orig_path = get_save_dir('training', 'frame') + f'/{file_name}.png',
        cv2.imwrite(mask_path, kwargs.get('orig', self.cropped_orig))
        cv2.imwrite(orig_path, kwargs.get('mask', self.annotated))

    # method runs in new thread
    def run(self):
        while not self.exit_flag:
            cv2.waitKey(1)  # waiting 1 ms speeds up UI side

            if self.export_frame_flag:
                export = self.cropped_orig.copy()
                export[self.annotated > 0] = self.annotated[self.annotated > 0]
                self.export_frame(export)
                self.export_frame_for_training(
                    orig=self.cropped_orig, 
                    mask=self.annotated)
                self.export_frame_flag = False

            # when new image is selected
            if self.url_updated_flag:
                self.orig_frame = MyFrame(cv2.imread(self.source_url), 'bgr')
                self.url_updated_flag = False

            if self.select_roi_flag and self.orig_frame is not None:
                r = cv2.selectROI("Select ROI", self.orig_frame)
                if all(r) != 0:
                    self.roi = r
                cv2.destroyWindow("Select ROI")
                self.roi_updated.emit(self.roi)
                self.select_roi_flag = False

            # get cropped frame whenever displaying frame
            if self.start_processing_flag or self.show_frame_flag:
                # get cropped frame
                if len(self.roi) > 0:
                    self.cropped_orig = self.orig_frame[
                        int(self.roi[1]):int(self.roi[1] + self.roi[3]),
                        int(self.roi[0]):int(self.roi[0] + self.roi[2])]
                else:
                    self.cropped_orig = self.orig_frame

                # start processing frame
                if self.start_processing_flag:
                    self.filtered = self.cropped_orig.copy()
                    self.annotated = MyFrame(
                        np.zeros(self.filtered.shape, dtype=np.uint8))

                    # probably make another flag for processing
                    while not self.q.empty():
                        p = self.q.get()
                        if p.__name__ == 'process':
                            self.filtered = p(self.filtered)
                        elif p.__name__ == 'analyze':
                            p(self.filtered)
                        elif p.__name__ == 'annotate':
                            self.annotated = p(self.annotated)
                    self.show_frame_flag = True

                if self.show_frame_flag:
                    show = MyFrame(
                        cv2.addWeighted(
                            self.cropped_orig.cvt_color('bgr'),
                            self.weight,
                            self.filtered.cvt_color('bgr'),
                            1 - self.weight,
                            1,
                        ), 'bgr')

                    show[self.annotated > 0] = self.annotated[
                        self.annotated > 0]
                    show = self.draw_cursor(show)
                    self.show_frame_flag = False

                    # cv2.imshow("Frame", frame)
                    rgbImage = show.cvt_color('rgb')
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    qt_img = QImage(rgbImage.data, w, h, bytesPerLine,
                                    QImage.Format_RGB888)
                    self.changePixmap.emit(qt_img)
                self.start_processing_flag = False

        cv2.destroyAllWindows()

    def update_cursor(self, x, y):
        self.cursor_x = x
        self.cursor_y = y

    def draw_cursor(self, frame):
        return MyFrame(
            cv2.circle(frame, (self.cursor_x, self.cursor_y), 1,
                       (255, 255, 255), -1))

    def stop(self):
        self.exit_flag = True
        self.wait()


# =---------------------------------------------------------


class DisplayFrame(QLabel):
    mouse_moved = pyqtSignal(int, int)
    mouse_pressed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.width = 800
        self.height = 600
        self.resize(self.width, self.height)
        self.setMouseTracking(True)

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.img = image
        self.show = image.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.setPixmap(QPixmap.fromImage(self.show))

        if self.parent is not None:
            self.parent.resize(self.show.width(), self.height)

    def mouseMoveEvent(self, event):
        scale = self.img.width() / self.show.width()
        x = event.x() * scale
        y = event.y() * scale
        self.mouse_moved.emit(int(x), int(y))
        self.parent.update_thread()

    def mousePressEvent(self, event):
        self.mouse_pressed.emit(event)
        self.parent.update_thread()


class BubbleAnalyzerWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bubble Analzyer")

        default_url = "source\\frame_init\\circle1.png"
        self.parameters = MyParams(default_url)
        self.parameters.paramChange.connect(self.on_param_change)
        self.init_ui()
        # connect mouse events to params
        # process images separate from ui
        self.thread = ImageProcessingThread(
            self,
            url=self.parameters.get_param_value("Settings", "File Select"),
            weight=self.parameters.get_param_value("Settings",
                                                   "Overlay Weight"),
            roi=self.parameters.internal_params["ROI"])
        self.thread.changePixmap.connect(self.display_label.set_image)
        self.thread.roi_updated.connect(self.on_roi_updated)
        self.display_label.mouse_moved.connect(self.thread.update_cursor)
        self.connect_param_signals()

        self.thread.start()
        self.update_thread()

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        self.layout = QHBoxLayout(self)
        # self.setFixedHeight(620)
        self.display_label = DisplayFrame(self)
        # self.plotter = CenterPlot()
        self.parameters.setFixedWidth(400)
        # align center to make sure mouse coordinates maps to what's shown
        self.layout.addWidget(self.display_label, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.parameters)
        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)

    def closeEvent(self, event):
        cv2.destroyAllWindows()
        self.thread.stop()
        self.parameters.save_settings()

    def on_roi_updated(self, roi):
        print('roi updated')
        self.parameters.internal_params["ROI"] = roi
        self.update_thread()

    def connect_param_signals(self):
        for param in self.parameters.params.child('Analyze').children():
            if isinstance(param, AnalyzeBubblesWatershed):
                self.display_label.mouse_moved.connect(
                    param.on_mouse_move_event)
                self.display_label.mouse_pressed.connect(
                    param.on_mouse_click_event)
                self.thread.roi_updated.connect(
                    param.on_roi_updated)

    # update frame canvas on param changes
    def on_param_change(self, parameter, changes):
        has_operation = False

        for param, change, data in changes:
            path = self.parameters.params.childPath(param)
            if path is None:
                continue
            if change == 'childRemoved':
                # on remove data is the object
                if isinstance(data, AnalyzeBubblesWatershed):
                    self.display_label.mouse_moved.disconnect(
                        data.on_mouse_move_event)
                    self.display_label.mouse_pressed.disconnect(
                        data.on_mouse_click_event)
                has_operation = True
                continue
            if change == 'childAdded':
                # on add data is a tuple containing object(s) added
                if isinstance(data[0], AnalyzeBubblesWatershed):
                    self.display_label.mouse_moved.connect(
                        data[0].on_mouse_move_event)
                    self.display_label.mouse_pressed.connect(
                        data[0].on_mouse_click_event)
                has_operation = True
                continue

            if path[0] == "Settings":
                if path[1] == "File Select":
                    self.thread.update_url(data)
                    self.parameters.update_url(data)
                    has_operation = True
                elif path[1] == "Overlay Weight":
                    self.thread.set_overlay_weight(data)
                elif path[1] == "Select ROI":
                    self.thread.get_roi()
            if path[0] == "Filter":
                has_operation = True
            if path[0] == "Analyze":
                has_operation = True
                if path[1] == "Bubbles":
                    if path[2] == "Export Distances":
                        self.thread.export_frame_flag = True
                if path[1] == 'BubbleWatershed':
                    if path[2] == 'export_csv' or path[
                            2] == 'export_graphs':
                        self.thread.export_frame_flag = True

        if has_operation:
            self.update_thread()

    def update_thread(self, filter=True, analyze=True, annotate=True):
        if filter:
            self.update_filters()
        if analyze:
            self.update_analysis()
        if annotate:
            self.update_annotations()
        if filter or analyze or annotate:
            # print(f'{self.thread.start_processing_flag=}')
            self.thread.start_image_operations()

    def update_filters(self):
        # op is a custom param object which contains the method process
        for op in self.parameters.params.child("Filter").children():    
            if op.child("Toggle").value():
                self.thread.add_to_queue(op.process)

    def update_analysis(self):
        # op is a custom param object which contains the method analyze
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value():
                self.thread.add_to_queue(op.analyze)

    def update_annotations(self):
        # draw on annotations at the very end
        # op is a custom param object which contains the method annotate
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value() and op.child("Overlay",
                                                       "Toggle").value():
                self.thread.add_to_queue(op.annotate)


def main():
    app = QApplication(sys.argv)
    main_window = BubbleAnalyzerWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
