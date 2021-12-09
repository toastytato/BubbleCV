import sys
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent
import cv2
import os
from queue import Queue
import traceback, sys

# --- my classes ---
from main_params import MyParams
from filters import *
from bubble_contour import get_save_dir
from misc_methods import MyFrame

# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
# - jk just make sure thread doesn't do things too quickly
# - by that I mean put a 1ms delay in infinite while loops


class ProcessingThread(QThread):
    changePixmap = pyqtSignal(QImage)
    url_updated = pyqtSignal()
    roi_updated = pyqtSignal(object)

    def __init__(self, parent, source_url, weight=0):
        super().__init__(parent)
        self.q = Queue(20)
        self.source_url = source_url
        self.split_url = os.path.splitext(source_url)
        self.orig_frame = None
        self.processed = None
        self.cropped = None
        self.roi = None
        self.weight = 0

        self.show_frame_flag = False
        self.exit_flag = False
        self.url_updated_flag = False
        self.processing_flag = False
        self.export_frame_flag = False
        self.select_roi_flag = False

    def update_url(self, url):
        self.source_url = url
        self.split_url = os.path.splitext(url)
        self.url_updated_flag = True
        self.processing_flag = True
        print("url updated")

    def get_roi(self):
        self.select_roi_flag = True
        self.processing_flag = True

    def set_weight(self, weight):
        print("W:", weight)
        self.weight = weight
        self.show_frame_flag = True

    def export_frame(self, frame):
        path = get_save_dir('analysis', self.source_url) + "/overlay.png"
        # if not os.path.exists("analysis/overlays"):
        #     os.makedirs("analysis/overlays")
        # name = os.path.basename(self.split_url[0])
        cv2.imwrite(path, frame)

    def run(self):

        while not self.exit_flag:
            cv2.waitKey(1)  # waiting 1 ms speeds up UI side

            if self.export_frame_flag:
                self.export_frame(self.processed)
                self.export_frame_flag = False

            # when new image is selected
            if self.url_updated_flag:
                self.orig_frame = MyFrame(cv2.imread(self.source_url), 'bgr')
                self.url_updated_flag = False

            # prevents adding to queue while processing
            if self.select_roi_flag and self.orig_frame is not None:
                r = cv2.selectROI("Select ROI", self.orig_frame)
                if all(r) != 0:
                    self.roi = r
                cv2.destroyWindow("Select ROI")
                self.roi_updated.emit(self.roi)
                self.select_roi_flag = False

            # get cropped frame whenever displaying frame
            if self.processing_flag or self.show_frame_flag:
                # get cropped frame
                if len(self.roi) > 0:
                    cropped_orig = self.orig_frame[
                        int(self.roi[1]):int(self.roi[1] + self.roi[3]),
                        int(self.roi[0]):int(self.roi[0] + self.roi[2])]
            # start processing frame
            if self.processing_flag:
                # probably make another flag for processing
                self.processed = cropped_orig.copy()
                cnt = 1
                while not self.q.empty():
                    p = self.q.get()
                    self.processed = p(self.processed)
                    print(f"{cnt}. {p.__name__}: {type(self.processed)}")
                    cnt += 1

                self.processing_flag = False
                self.show_frame_flag = True

            if self.show_frame_flag:
                cropped_orig = cv2.addWeighted(
                    cropped_orig.cvt_color('bgr'),
                    self.weight,
                    self.processed.cvt_color('bgr'),
                    1 - self.weight,
                    1,
                )
                self.show_frame_flag = False
                # frame = self.processed

                # cv2.imshow("Frame", frame)
                rgbImage = cv2.cvtColor(cropped_orig, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine,
                                           QImage.Format_RGB888)
                p = convertToQtFormat.scaled(800, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                # cv2.waitKey(1)

    def stop(self):
        cv2.destroyAllWindows()
        self.exit_flag = True
        self.wait()


# =---------------------------------------------------------


class BubbleAnalyzerWindow(QMainWindow):

    thread_update_queue = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bubble Analzyer")
        default_url = "source\\frame_init\\circle1.png"

        self.parameters = MyParams(default_url)
        url = self.parameters.get_param_value("Settings", "File Select")
        self.parameters.update_url(url)

        self.init_ui()

        self.thread = ProcessingThread(self, source_url=default_url, weight=0)
        self.thread.changePixmap.connect(self.set_image)
        self.thread.roi_updated.connect(self.on_roi_updated)
        self.thread.update_url(default_url)

        self.parameters.paramChange.connect(self.on_param_change)

        self.thread.start()
        self.thread.roi = self.parameters.internal_params["ROI"]
        self.thread.weight = self.parameters.get_param_value(
            "Settings", "Overlay Weight")
        self.send_to_thread()

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        self.layout = QHBoxLayout(self)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(620, 620)
        # create a video label
        self.video_label = QLabel(self)
        # self.video_label.move(280, 120)
        # self.video_label.resize(720, 640)

        # self.plotter = CenterPlot()
        self.parameters.setFixedWidth(400)
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.parameters)
        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)

    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, a0: QCloseEvent):
        cv2.destroyAllWindows()
        self.thread.stop()
        self.parameters.save_settings()
        return super().closeEvent(a0)

    def on_roi_updated(self, roi):
        self.parameters.internal_params["ROI"] = self.thread.roi
        self.send_to_thread()

    def on_file_selected(self, url):
        print(url)

    def on_param_change(self, parameter, changes):
        has_operation = False

        if self.thread.processing_flag:
            print("thread busy, ignoring param change")
            return

        for param, change, data in changes:
            if change == 'childRemoved' or change == 'childAdded':
                has_operation = True
                continue
            path = self.parameters.params.childPath(param)
            if path is None:
                continue

            if path[0] == "Settings":
                if path[1] == "File Select":
                    self.thread.update_url(data)
                    self.parameters.update_url(data)
                    has_operation = True
                elif path[1] == "Overlay Weight":
                    self.thread.set_weight(data)
                elif path[1] == "Select ROI":
                    self.thread.get_roi()
            if path[0] == "Filter":
                has_operation = True
            if path[0] == "Analyze":
                has_operation = True
                if path[1] == "Bubbles":
                    if path[2] == "Export Distances":
                        self.thread.export_frame_flag = True

        if has_operation:
            self.send_to_thread()

        # todo: make sure thread does all the processing first and then overlay

    def send_to_thread(self):
        # filter incoming image
        for op in self.parameters.params.child("Filter").children():
            if op.child("Toggle").value():
                self.thread.q.put(op.process)

        # do all necessary processing without manipulating image
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value():
                self.thread.q.put(op.process)

        # draw on annotations at the very end
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value() and op.child("Overlay",
                                                       "Toggle").value():
                self.thread.q.put(op.annotate)

        self.thread.processing_flag = True


def main():
    app = QApplication(sys.argv)
    main_window = BubbleAnalyzerWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
