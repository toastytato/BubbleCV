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

# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
# - jk just make sure thread doesn't do things too quickly
# - by that I mean put a 1ms delay in infinite while loops


class ProcessingThread(QThread):
    changePixmap = pyqtSignal(QImage)
    finished = pyqtSignal(bool)
    url_updated = pyqtSignal()
    roi_updated = pyqtSignal(object)

    def __init__(self, parent, source_url, weight=0):
        super().__init__(parent)
        self.q = Queue(20)
        self.source_url = source_url
        self.split_url = os.path.splitext(source_url)
        self.orig_frame = None
        self.processed = None
        self.roi = None
        self.weight = 0

        self.exit_flag = False
        self.url_updated_flag = False
        self.resume_flag = False
        self.export_frame_flag = False
        self.select_roi_flag = False
        self.changing_just_weight_flag = False

    def update_url(self, url):
        self.source_url = url
        self.split_url = os.path.splitext(url)
        self.url_updated_flag = True
        self.resume_flag = True 
        self.url_updated.emit()
        print("url updated")

    def get_roi(self):
        self.select_roi_flag = True
        self.resume_flag = True

    def set_weight(self, weight):
        print("W:", weight)
        self.weight = weight
        self.changing_just_weight_flag = True
        self.resume_flag = True

    def export_frame(self, frame):
        if not os.path.exists('analysis/overlays'):
            os.makedirs('analysis/overlays')
        name = os.path.basename(self.split_url[0])
        cv2.imwrite("analysis/overlays/" + name + "_overlay.png", frame)


    def run(self):

        while not self.exit_flag:
            cv2.waitKey(1)  # waiting 1 ms speeds up UI side

            if self.export_frame_flag:
                self.export_frame(self.processed)
                self.export_frame_flag = False

            # for processing images
            if self.resume_flag:
                # prevents adding to queue while processing
                self.finished.emit(False)

                if self.orig_frame is None or self.url_updated_flag:
                    frame = cv2.imread(self.source_url)
                    self.url_updated_flag = True

                if self.select_roi_flag:
                    r = cv2.selectROI("Select ROI", frame)
                    if all(r) != 0:
                        self.roi = r
                    cv2.destroyWindow("Select ROI")
                    self.select_roi_flag = False
                    self.roi_updated.emit(self.roi)

                if len(self.roi) > 0:
                    frame = frame[
                        int(self.roi[1]) : int(self.roi[1] + self.roi[3]),
                        int(self.roi[0]) : int(self.roi[0] + self.roi[2]),
                    ]

                # probably make another flag for processing
                if not self.changing_just_weight_flag:
                    self.processed = frame.copy()
                    while not self.q.empty():
                        p = self.q.get()
                        try:
                            self.processed = p(frame=self.processed)
                        except Exception as e:
                            print(e)
                            break
                self.changing_just_weight_flag = False

                frame = cv2.addWeighted(
                    frame,
                    self.weight,
                    self.processed,
                    1 - self.weight,
                    1,
                )
                # frame = self.processed

                # cv2.imshow("Frame", frame)
                # cv2.waitKey(1)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
                )
                p = convertToQtFormat.scaled(800, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.resume_flag = False
                self.finished.emit(True)
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
        url = "Frame\cluster3_18f_orig.png"

        self.parameters = MyParams(url)

        self.init_ui()

        self.ready = True
        url = self.parameters.get_param_value("Settings", "File Select")
        self.thread = ProcessingThread(self, source_url=url, weight=0)
        self.thread.changePixmap.connect(self.set_image)
        self.thread.finished.connect(self.thread_ready)
        self.thread.url_updated.connect(self.send_to_thread)

        self.parameters.paramChange.connect(self.on_param_change)

        self.thread.start()
        self.thread.roi = self.parameters.internal_params["ROI"]
        self.thread.resume_flag = True
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
        self.thread.stop()
        self.parameters.internal_params["ROI"] = self.thread.roi
        self.parameters.save_settings()
        return super().closeEvent(a0)

    def thread_ready(self, state):
        self.ready = state

    def on_file_selected(self, url):
        print(url)

    def on_param_change(self, parameter, changes):
        has_operation = False

        for param, change, data in changes:
            path = self.parameters.params.childPath(param)
            print("Path:", path)

            if path is None:
                break
            if path[0] == "Settings":
                if path[1] == "File Select":
                    self.thread.update_url(data)
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

        if has_operation and self.ready:
            self.send_to_thread()

        # todo: make sure thread does all the processing first and then overlay

    def send_to_thread(self):
        # print("main:", self.id())

        # filter incoming image
        for op in self.parameters.params.child("Filter").children():
            if op.child("Toggle").value():
                # self.thread_update_queue.emit(op.process)
                self.thread.q.put(op.process)
        # do all necessary processing without manipulating image
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value():
                self.thread.q.put(op.process)

                # self.thread_update_queue.emit(op.process)
        # draw on annotations at the very end
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value() and op.child("Overlay", "Toggle").value():
                self.thread.q.put(op.annotate)

        # self.thread_update_queue.emit(op.annotate)
        self.thread.resume_flag = True
        # self.thread.start_flag = True


def main():
    app = QApplication(sys.argv)
    main_window = BubbleAnalyzerWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
