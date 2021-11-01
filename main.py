import sys
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
import imutils
from matplotlib.pyplot import contour
from numpy import source
from pyqtgraph import parametertree, plot
from pyqtgraph.colormap import *
import time
import csv

# --- my classes ---
from parameters import MyParams
from my_cv_process import *
from plotter import CenterPlot

source_url = "Processed\\frame1.png"


class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent, parameters):
        super().__init__(parent)
        self.parameters = parameters
        self.exit = False
        self.update_filter_flag = True
        self.update_overlay_flag = True
        self.analyed_flag = False

    def run(self):

        extension = os.path.splitext(source_url)[1]
        if extension == ".png":
            frame = cv2.imread(source_url)
            while True:
                if self.update_filter_flag:
                    self.process(frame)
                    self.update_filter_flag = False
                if self.update_overlay_flag:
                    # self.update(frame)
                    self.update_overlay_flag = False

                if self.exit:
                    break

        elif extension == ".mp4":
            cap = cv2.VideoCapture(source_url)
            width = int(cap.get(3))
            height = int(cap.get(4))
            out = cv2.VideoWriter(
                "output.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 30.0, (width, height)
            )
            framerate = cap.get(cv2.CAP_PROP_FPS) / 2
            prev = 0
            frame_counter = 0

            while cap.isOpened():
                time_elapsed = time.time() - prev
                if time_elapsed > 1 / framerate:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_counter += 1
                    prev = time.time()
                    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        frame_counter = 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    self.process(frame)

                if self.exit:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows
            print("Video Saved")

    def process(self, frame):
        print("Processing")
        # frame = imutils.resize(frame, width=900)
        self.processed = frame.copy()
        # if self.parameters.get_param_value("Threshold", "Toggle"):
        # mask = thresh(
        #     frame=mask,
        #     lower=self.parameters.get_param_value("Threshold", "Lower"),
        #     upper=self.parameters.get_param_value("Threshold", "Upper"),
        # )
        if self.parameters.get_param_value("Invert", "Toggle"):
            self.processed = invert(
                frame=self.processed,
            )

        if self.parameters.get_param_value("Blob Filter", "Toggle"):
            self.processed = dilate(
                frame=self.processed,
                erode=self.parameters.get_param_value("Blob Filter", "Erode"),
                dilate=self.parameters.get_param_value("Blob Filter", "Dilate"),
            )

        if self.parameters.get_param_value("Analyze Circles", "Toggle"):
            self.bubbles = contours(
                frame=self.processed,
                min=self.parameters.get_param_value("Analyze Circles", "Min Size"),
            )
            distances, neighbor_idx = get_neighbor_distances(
                bubbles=self.bubbles,
            )
            self.analyed_flag = True

        # draw annotations
        if self.parameters.get_param_value("Overlay", "Toggle") and self.analyed_flag:
            self.processed = draw_annotations(
                frame=self.processed,
                bubbles=self.bubbles,
                neighbor_idx=neighbor_idx,
                distances=distances,
                offset=self.parameters.get_param_value(
                    "Analyze Circles", "Bounds Offset"
                ),
                highlight_idx=self.parameters.get_param_value(
                    "Overlay", "Bubble Highlight"
                ),
                center_color=self.parameters.get_param_value(
                    "Overlay", "Center Color"
                ).name(),
                circum_color=self.parameters.get_param_value(
                    "Overlay", "Circumference Color"
                ).name(),
                highlight_color=self.parameters.get_param_value(
                    "Overlay", "Highlight Color"
                ).name(),
            )
            self.analyed_flag = False

        # frame[np.where((frame > [150, 150, 150]).all(axis=2))] = [0, 0, 150]
        weight = self.parameters.get_param_value("Overlay", "Mask Weight")
        frame = cv2.addWeighted(frame, weight, self.processed, 1 - weight, 1)

        rgbImage = cv2.cvtColor(self.processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(
            rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
        )
        p = convertToQtFormat.scaled(800, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)

    def update(self, frame):
        pass

    def stop(self):
        self.exit = True
        self.wait()


class BubbleAnalyzerWindow(QMainWindow):
    slider_value = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bubble Analzyer")
        self.parameters = MyParams()

        self.init_ui()

        self.vid_thread = VideoThread(self, self.parameters)
        self.vid_thread.start()
        self.vid_thread.changePixmap.connect(self.set_image)
        self.parameters.paramChange.connect(self.on_param_change)

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        self.layout = QHBoxLayout(self)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1000, 480)
        # create a label
        self.video_label = QLabel(self)
        self.video_label.move(280, 120)
        self.video_label.resize(640, 480)

        # self.plotter = CenterPlot()

        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.parameters)
        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.vid_thread.stop()
        self.parameters.save_settings()

    def on_param_change(self, parameter, changes):
        print("param changed")
        self.vid_thread.update_filter_flag = True
        for param, change, data in changes:
            path = self.parameters.params.childPath(param)

            if path[0] == "Overlay":
                if path[1] == "Export Distances":
                    print("exporting dist")
                    # export_csv(self.vid_thread.bubbles)
        #     elif path[0] == "Blob Filter":
        #         if path[1] == "Erode":
        #             self.vid_thread.processor.erode_iterations = data
        #         elif path[1] == "Dilate":
        #             self.vid_thread.processor.dilate_iterations = data
        #     elif path[0] == "Tracker":
        #         pass


def main():
    app = QApplication(sys.argv)
    main_window = BubbleAnalyzerWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
