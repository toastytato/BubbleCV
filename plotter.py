
import pyqtgraph as pg


# Graph Widget
class CenterPlot(pg.PlotWidget):
    def __init__(self, legend=None):
        super().__init__()
        # PlotWidget super functions
        self.line_width = 1
        self.legend = legend
        self.num_circles = 5
        self.x_accum = [[0] for i in range(self.num_circles)]
        self.y_accum = [[0] for i in range(self.num_circles)]
        self.curr_buffer_size = 0
        self.buffer_index = 0
        self.buffer_limit = 100

        self.frame_cnt = 0

        if self.legend is None:  # default colors if no legend has been created
            self.curve_colors = ["b", "g", "r", "c", "y", "m"]
        else:
            self.curve_colors = self.legend.curve_colors

        self.pens = [pg.mkPen(i, width=self.line_width) for i in self.curve_colors]
        self.showGrid(y=True)
        # self.disableAutoRange('y')

        self.setLabel("left", "Voltage", units="V")
        self.setLabel("bottom", "Samples")

    def update_logs(self, incoming_data):
        self.frame_cnt += 1

        for i in range(self.num_circles):
            if self.curr_buffer_size < self.buffer_limit:
                try:
                    self.x_accum[i].append(incoming_data[i][0])
                    self.y_accum[i].append(incoming_data[i][1])
                except IndexError:
                    print(len(incoming_data))
            else:
                self.x_accum[i][self.buffer_index] = incoming_data[i][0]
                self.y_accum[i][self.buffer_index] = incoming_data[i][1]
                self.buffer_index = (self.buffer_index + 1) % self.buffer_limit

        self.curr_buffer_size += 1

        # adds each channel's offset to incoming channel

    def update_plot(self):
        self.clear()
        for i, data in enumerate(self.x_accum):
            #     # this is kinda ugly but it works so let's roll with it
            self.plot(data, clear=False, pen=self.pens[i])
