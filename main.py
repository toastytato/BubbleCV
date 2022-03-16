from cmath import inf
from email.policy import default
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

import matplotlib.pyplot as plt

# --- my classes ---
from main_params import RESET_DEFAULT_PARAMS, MyParams
from filters import *
from bubble_analysis import get_save_dir
from analysis_params import Analysis
from video_thread import ImageProcessingThread
# =---------------------------------------------------------


# main ui, handles controller logic
class BubbleAnalyzerWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bubble Analyzer")

        default_url = "source\\022622_bubbleformation_80x_075M_3_original.avi"
        self.parameters = MyParams(default_url)
        self.parameters.paramChange.connect(self.on_param_change)

        self.init_ui()

        analysis = self.parameters.get_child('BubbleWatershed')

        # process images separate from ui
        self.cv_thread = ImageProcessingThread(
            self,
            url=default_url,
            analysis=analysis)
        # self.cv_thread.changePixmap.connect(self.display_label.set_image)
        self.cv_thread.view_frame.connect(self.display_view)
        self.cv_thread.start()

    @pyqtSlot(str, object)
    def display_view(self, title, frame):
        # plt.figure()
        # plt.imshow(frame)
        cv2.imshow(title, frame)

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        # self.setFixedHeight(620)
        self.layout = QGridLayout(self)

        # display_layout.addWidget(self.display_timeline)
        # self.plotter = CenterPlot()
        self.parameters.setFixedWidth(400)
        # align center to make sure mouse coordinates maps to what's shown
        # self.layout.addWidget(self.display_label, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.parameters, 0, 1, 2, 1, Qt.AlignCenter)

        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)
        # self.resize(self.minimumSizeHint())

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
