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
from matplotlib.pyplot import contour
from numpy import source
from queue import Queue
import traceback, sys

# --- my classes ---
from main_params import MyParams
from my_cv_process import *
from plotter import CenterPlot

# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
#


class ProcessingThread(QThread):
    changePixmap = pyqtSignal(QImage)
    finished = pyqtSignal(bool)

    def __init__(self, parent, source_url, *args, **kwargs):
        super().__init__(parent)
        self.q = Queue(20)
        self.source_url = source_url
        self.processed = None
        self.exit_flag = False
        self.args = args
        self.kwargs = kwargs
        self.start_flag = True

    def start_processing(self):
        self.start_flag = True

    def update_filter(self, op):
        self.q.put(op)

    def run(self):

        extension = os.path.splitext(self.source_url)[1]
        while not self.exit_flag:
            # for processing images
            cv2.waitKey(1)  # waiting 1 ms speeds up UI side
            if self.start_flag:
                # prevents adding to queue while processing
                self.finished.emit(False)

                frame = cv2.imread(self.source_url)
                self.processed = frame.copy()
                print("Thread Processing")
                while not self.q.empty():
                    print("Reading Queue")
                    p = self.q.get()
                    self.processed = p(frame=self.processed)

                frame = cv2.addWeighted(
                    frame,
                    self.kwargs["weight"],
                    self.processed,
                    1 - self.kwargs["weight"],
                    1,
                )

                # cv2.imshow("TItle", frame)
                # cv2.waitKey(1)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
                )
                p = convertToQtFormat.scaled(800, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.start_flag = False
                self.finished.emit(True)
                # cv2.waitKey(1)

    def stop(self):
        self.exit_flag = True
        self.wait()


# =---------------------------------------------------------


class BubbleAnalyzerWindow(QMainWindow):

    thread_update_queue = pyqtSignal(object)
    thread_start_flag = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bubble Analzyer")
        self.parameters = MyParams()

        self.init_ui()

        self.ready = True
        url = "Processed\\frame1.png"
        self.thread = ProcessingThread(self, source_url=url, weight=0)
        self.thread.changePixmap.connect(self.set_image)
        self.thread.finished.connect(self.thread_ready)
        self.thread_update_queue.connect(self.thread.update_filter)
        self.thread_start_flag.connect(self.thread.start_processing)

        self.parameters.paramChange.connect(self.on_param_change)

        # self.start_flag.emit()
        self.thread.start()
        self.send_to_thread()

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        self.layout = QHBoxLayout(self)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(960, 640)
        # create a video label
        self.video_label = QLabel(self)
        self.video_label.move(280, 120)
        self.video_label.resize(720, 640)

        # self.plotter = CenterPlot()

        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.parameters)
        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)

    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, a0: QCloseEvent):
        self.thread.stop()
        self.parameters.save_settings()
        return super().closeEvent(a0)

    def thread_ready(self, state):
        self.ready = state

    def on_param_change(self, parameter, changes):
        has_operation = False

        for param, change, data in changes:
            path = self.parameters.params.childPath(param)
            if path is None:
                break
            if path[0] == "Filter":
                has_operation = True
            if path[0] == "Analyze":
                has_operation = True

        if has_operation and self.ready:
            self.send_to_thread()
            # q.put(op.process)
        # print("Main q id:", id(q))
        # self.thread_update_queue.emit(q)
        # todo: make sure thread does all the processing first and then overlay

    def send_to_thread(self):
        # filter incoming image
        for op in self.parameters.params.child("Filter").children():
            if op.child("Toggle").value():
                self.thread_update_queue.emit(op.process)
        # do all necessary processing without manipulating image
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value():
                self.thread_update_queue.emit(op.process)
        # draw on annotations at the very end
        for op in self.parameters.params.child("Analyze").children():
            if op.child("Toggle").value() and op.child("Overlay", "Toggle").value():
                self.thread_update_queue.emit(op.annotate)
        # self.thread_start_flag.emit()
        self.thread.start_flag = True


def main():
    app = QApplication(sys.argv)
    main_window = BubbleAnalyzerWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

    # # for processing videos
    # elif extension == ".mp4":
    #     cap = cv2.VideoCapture(source_url)
    #     width = int(cap.get(3))
    #     height = int(cap.get(4))
    #     out = cv2.VideoWriter(
    #         "output.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 30.0, (width, height)
    #     )
    #     framerate = cap.get(cv2.CAP_PROP_FPS) / 2
    #     prev = 0
    #     frame_counter = 0

    #     while cap.isOpened():
    #         time_elapsed = time.time() - prev
    #         if time_elapsed > 1 / framerate:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #             frame_counter += 1
    #             prev = time.time()
    #             if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #                 frame_counter = 0
    #                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #             self.process(frame)

    #         if self.exit:
    #             break

    #     cap.release()
    #     out.release()
    #     cv2.destroyAllWindows
    #     ("Video Saved")
