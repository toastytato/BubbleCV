import sys
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import (
    QThread,
    QThreadPool,
    Qt,
    pyqtSignal,
    QRunnable,
    QObject,
)
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent
import cv2
import os
import imutils
from matplotlib.pyplot import contour
from numpy import source
import time
import queue
import traceback, sys

# --- my classes ---
from parameters import MyParams
from my_cv_process import *
from plotter import CenterPlot

source_url = "Processed\\circle1.png"

# thread notes:
# - only use signals
# - don't use global flags: will only slow down main thread
# 

class ProcessingThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)
        self.q = queue.Queue(10)
        self.processed = None
        self.exit_flag = False
        self.process_flag = True
        self.args = args
        self.kwargs = kwargs

    def update_filter(self, process):
        self.q.put(process)

    def run(self):

        extension = os.path.splitext(source_url)[1]
        while not self.exit_flag:
            # for processing images
            if extension == ".png":

                frame = cv2.imread(source_url)
                self.processed = frame.copy()
                p = self.q.get()
                while p != "done":
                    self.processed = p(frame=self.processed)
                    p = self.q.get()

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

    # def set_operation(self, operation):
    #     self.update_filter_flag = True
    #     self.operation = operation

    # def process(self, frame):
    #     print("Processing")
    #     # frame = imutils.resize(frame, width=900)

    #     if self.parameters.get_param_value("Analyze Circles", "Toggle"):
    #         self.bubbles = get_contours(
    #             frame=self.processed,
    #             min=self.parameters.get_param_value("Analyze Circles", "Min Size"),
    #         )
    #         min, max = get_bounds(
    #             bubbles=self.bubbles,
    #             offset=self.parameters.get_param_value(
    #                 "Analyze Circles", "Bounds Offset"
    #             ),
    #         )
    #         get_neighbors(
    #             bubbles=self.bubbles,
    #             num_neighbors=self.parameters.get_param_value(
    #                 "Analyze Circles", "Num Neighbors"
    #             ),
    #         )
    #         self.analyed_flag = True

    #     # draw annotations
    #     if self.parameters.get_param_value("Overlay", "Toggle") and self.analyed_flag:
    #         self.processed = draw_annotations(
    #             frame=self.processed,
    #             bubbles=self.bubbles,
    #             min=min,
    #             max=max,
    #             highlight_idx=self.parameters.get_param_value(
    #                 "Overlay", "Bubble Highlight"
    #             ),
    #             center_color=self.parameters.get_param_value(
    #                 "Overlay", "Center Color"
    #             ).name(),
    #             circum_color=self.parameters.get_param_value(
    #                 "Overlay", "Circumference Color"
    #             ).name(),
    #             neighbor_color=self.parameters.get_param_value(
    #                 "Overlay", "Neighbor Color"
    #             ).name(),
    #         )
    #         self.analyed_flag = False

    #     # frame[np.where((frame > [150, 150, 150]).all(axis=2))] = [0, 0, 150]
    #     weight = self.parameters.get_param_value("Overlay", "Mask Weight")
    #     frame = cv2.addWeighted(frame, weight, self.processed, 1 - weight, 1)

    #     rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgbImage.shape
    #     bytesPerLine = ch * w
    #     convertToQtFormat = QImage(
    #         rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
    #     )
    #     p = convertToQtFormat.scaled(800, 480, Qt.KeepAspectRatio)
    #     self.changePixmap.emit(p)

    # def update(self, frame):
    #     pass

    def stop(self):
        self.exit_flag = True
        self.wait()


# =---------------------------------------------------------


class BubbleAnalyzerWindow(QMainWindow):

    update_filter = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bubble Analzyer")
        self.parameters = MyParams()

        self.init_ui()

        self.thread = ProcessingThread(self, weight=0)
        self.update_filter.connect(self.thread.update_filter)
        self.thread.changePixmap.connect(self.set_image)
        self.thread.start()

        self.parameters.paramChange.connect(self.on_param_change)

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        self.layout = QHBoxLayout(self)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1000, 480)
        # create a video label
        self.video_label = QLabel(self)
        self.video_label.move(280, 120)
        self.video_label.resize(640, 480)

        # self.plotter = CenterPlot()

        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.parameters)
        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)

    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, a0: QCloseEvent):
        # self.vid_thread.stop()
        # self.parameters.save_settings()
        return super().closeEvent(a0)

    def on_param_change(self, parameter, changes):

        self.parameters.paramChange.disconnect(self.on_param_change)
        for param, change, data in changes:
            path = self.parameters.params.childPath(param)
            print(param)
            if path is None:
                break
            if path[0] == "Filters":
                for filter in self.parameters.params.child("Filters").children():
                    if filter.child("Toggle").value():
                        self.update_filter.emit(filter.process)
                self.update_filter.emit("done")

            if path[0] == "Overlay":
                if path[1] == "Mask Weight":
                    print(data)


            # if filter.child("Toggle").value():
            #     filter.moveToThread(self.thread)
            #     self.thread.started.connect(filter.process)
            #     filter.finished.connect(self.thread.quit)
        self.parameters.paramChange.connect(self.on_param_change)


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
    #     print("Video Saved")
