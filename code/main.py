import sys
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QGridLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QApplication,
    QMainWindow,
)
from PyQt5.QtCore import (
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5 import QtGui
import cv2
import sys
import matplotlib.pyplot as plt
import pyqtgraph as pg

# --- my classes ---
from main_params import RESET_DEFAULT_PARAMS, MyParams
from filters import *
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
        self.imv = pg.ImageView(view=pg.PlotItem())

        self.init_ui()

        analysis_params = self.parameters.params.children()[0]
        # process images separate from ui
        self.cv_thread = ImageProcessingThread(
            imageView=self.imv,
            analysis=analysis_params)  # , width=1080)
        # self.cv_thread.view_frame.connect(self.display_view)
        self.cv_thread.start()

    # need to display cv2.imshow in main thread else crash
    @pyqtSlot(str, object)
    def display_view(self, title, frame):
        # cv2.imshow(title, frame)
        self.imv.setImage(frame,
                          autoRange=False,
                          autoHistogramRange=False,
                          axes={
                              'y': 0,
                              'x': 1,
                              'c': 2
                          })

        value = self.imv.cursor().pos()
        print(value)

    @pyqtSlot(str, object)
    def plot_view(self, title, frame):
        fig, ax = plt.subplots()
        ax.imshow(frame)``
        ax.set_title(title)
        plt.show()

    def init_ui(self):
        self.mainbox = QWidget(self)
        self.setCentralWidget(self.mainbox)
        # self.setFixedHeight(620)
        self.layout = QGridLayout(self)

        # display_layout.addWidget(self.display_timeline)
        # self.plotter = CenterPlot()
        self.parameters.setFixedWidth(400)
        # align center to make sure mouse coordinates maps to what's shown
        self.layout.addWidget(self.parameters, 0, 1)
        self.layout.addWidget(self.imv, 0, 0, 4, 1)

        # self.layout.addWidget(self.plotter)
        self.mainbox.setLayout(self.layout)
        # self.resize(self.minimumSizeHint())

    # update frame canvas on param changes
    @pyqtSlot(object, object)
    def on_param_change(self, parameter, changes):
        # print('changed')
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
