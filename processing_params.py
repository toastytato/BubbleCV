from matplotlib.pyplot import gray, xscale
from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

import cv2
import numpy as np

### my classes ###
from bubble_contour import *
from filters import my_threshold, my_invert
from misc_methods import MyFrame, register_my_param


class Process(Parameter):

    def __init__(self, **opts):
        opts['removable'] = True
        super().__init__(**opts)
        # self.sigRemoved.connect(self.on_removed)

    def process(self, frame):
        return frame

    def annotate(self, frame):
        return frame

    def __repr__(self):
        msg = self.opts['name'] + ' Process'
        for c in self.childs:
            msg += f'\n{c.name()}: {c.value()}'
        return msg


@register_my_param
class AnalyzeBubbles(Process):
    cls_type = 'Bubbles'

    def __init__(self, url, **opts):
        opts['url'] = url
        if 'name' not in opts:
            opts['name'] = 'Bubbles'
        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': True
                },
                {
                    'name': 'Min Size',
                    'type': 'slider',
                    'value': 50,
                    'limits': (0, 200),
                },
                {
                    'name': 'Num Neighbors',
                    'type': 'int',
                    'value': 4
                },
                {
                    'name': 'Bounds Offset X',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Bounds Offset Y',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Bounds Scale X',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Bounds Scale Y',
                    'type': 'slider',
                    'value': 0,
                    'step': 1,
                    'limits': (-200, 200),
                },
                {
                    'name': 'Conversion',
                    'type': 'float',
                    'units': 'um/px',
                    'value': 600 / 900,
                    'readonly': True,
                },
                {
                    'name': 'Export Distances',
                    'type': 'action'
                },
                {
                    'name': 'Export Graph',
                    'type': 'action'
                },
                {
                    'name':
                    'Overlay',
                    'type':
                    'group',
                    'children': [
                        {
                            'name': 'Toggle',
                            'type': 'bool',
                            'value': True
                        },
                        {
                            'name': 'Bubble Highlight',
                            'type': 'int',
                            'value': 0
                        },
                        {
                            'name': 'Center Color',
                            'type': 'color',
                            'value': '#ff0000',
                        },
                        {
                            'name': 'Circumference Color',
                            'type': 'color',
                            'value': '#2CE2EE',
                        },
                        {
                            'name': 'Neighbor Color',
                            'type': 'color',
                            'value': '#2C22EE',
                        },
                    ],
                },
            ]
        super().__init__(**opts)
        self.bubbles = []
        self.url = url
        self.um_per_pixel = self.child('Conversion').value()
        self.child('Export Distances').sigActivated.connect(self.export_csv)
        self.child('Export Graph').sigActivated.connect(self.export_graphs)

        # self.sigTreeStateChanged.connect(self.on_change)

    def export_csv(self, change):
        # print('Export', change)

        if self.bubbles is not None:
            if self.url is None:
                export_csv(  # from bubble_processes
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url='exported_data',
                )
                print('Default Export')
            else:
                export_csv(
                    bubbles=self.bubbles,
                    conversion=self.um_per_pixel,
                    url=self.url + '_data',
                )

    def export_graphs(self, change):
        print(self.url)
        export_boxplots(
            self.bubbles,
            self.child('Num Neighbors').value(),
            self.um_per_pixel,
            self.url,
        )
        export_scatter(
            self.bubbles,
            self.child('Num Neighbors').value(),
            self.um_per_pixel,
            self.url,
        )
        export_dist_histogram(self.bubbles,
                              self.child('Num Neighbors').value(),
                              self.um_per_pixel, self.url)
        export_diam_histogram(self.bubbles,
                              self.child('Num Neighbors').value(),
                              self.um_per_pixel, self.url)

    def process(self, frame):
        self.bubbles = get_contours(frame=frame,
                                    min=self.child('Min Size').value())
        if len(self.bubbles) > self.child('Num Neighbors').value():
            self.lower_bound, self.upper_bound = get_bounds(
                bubbles=self.bubbles,
                scale_x=self.child('Bounds Scale X').value(),
                scale_y=self.child('Bounds Scale Y').value(),
                offset_x=self.child('Bounds Offset X').value(),
                offset_y=self.child('Bounds Offset Y').value(),
            )
            get_neighbors(bubbles=self.bubbles,
                          num_neighbors=self.child('Num Neighbors').value()
                          )  # modifies param to assign neighbors to bubbles
        return frame

    def annotate(self, frame):
        try:
            return draw_annotations(
                frame=frame.cvt_color('bgr'),
                bubbles=self.bubbles,
                min=self.lower_bound,
                max=self.upper_bound,
                highlight_idx=self.child('Overlay',
                                         'Bubble Highlight').value(),
                circum_color=self.child('Overlay',
                                        'Circumference Color').value(),
                center_color=self.child('Overlay', 'Center Color').value(),
                neighbor_color=self.child('Overlay', 'Neighbor Color').value(),
            )
        except AttributeError:
            return frame


@register_my_param
class AnalyzeBubblesWatershed(Process):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = 'BubblesWatershed'

    def __init__(self, url, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type
        opts['url'] = url

        self.img = {
            'gray': None,
            'thresh': None,
            'bg': None,
            'fg': None,
            'sel': None,
            'dist': None,
            'unknown': None,
            'final': None,
        }

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'BubbleWatershed'
        if 'children' not in opts:
            opts['children'] = [{
                'name': 'Toggle',
                'type': 'bool',
                'value': True
            }, {
                'name': 'Upper',
                'type': 'slider',
                'value': 255,
                'limits': (0, 255)
            }, {
                'name': 'Lower',
                'type': 'slider',
                'value': 0,
                'limits': (0, 255)
            }, {
                'title': 'FG scale',
                'name': 'fg_scale',
                'type': 'slider',
                'value': 0.01,
                'precision': 4,
                'step': 0.0005,
                'limits': (0, 1),
            }, {
                'title': 'Adapt Size',
                'name': 'adaptive',
                'type': 'slider',
                'value': 1,
                'step': 1,
                'limits': (1, 255),
            }, {
                'title': 'BG Iterations',
                'name': 'bg_iter',
                'type': 'int',
                'value': 3,
                'limits': (0, 255),
            }, {
                'title': 'Dist Transform Iter.',
                'name': 'dist_iter',
                'type': 'list',
                'value': 5,
                'limits': [0, 3, 5],
            }, {
                'title': 'Manual Select',
                'name': 'manual_sel',
                'type': 'action',
            }, {
                'name':
                'Overlay',
                'type':
                'group',
                'children': [
                    {
                        'name': 'Toggle',
                        'type': 'bool',
                        'value': True
                    },
                    {
                        'name': 'Toggle Text',
                        'type': 'bool',
                        'value': False
                    },
                    {
                        'title': 'View List',
                        'name': 'view_list',
                        'type': 'list',
                        'value': list(self.img.keys())[-1],
                        'limits': list(self.img.keys()),
                    },
                ]
            }]
        super().__init__(**opts)

        # manual sel states:
        self.sel_window_title = "Manual Selection"
        self.x = 0
        self.y = 0
        self.annotated = None
        self.prev_window_state = False
        self.show_selection_window = False
        self.mouse_update = False
        self.draw_flag = False

        self.child('manual_sel').sigActivated.connect(
            self.on_manual_selection_clicked)

    # video thread
    def process(self, frame):
        print('start processing')
        # frame = MyFrame(frame, 'bgr')
        self.img['gray'] = frame.cvt_color('gray')

        _, self.img['thresh'] = cv2.threshold(self.img['gray'],
                                              self.child('Lower').value(),
                                              self.child('Upper').value(),
                                              cv2.THRESH_BINARY_INV)
        self.img['thresh'] = MyFrame(self.img['thresh'], 'gray')
        # expanded threshold to indicate outer bounds of interest
        kernel = np.ones((3, 3), np.uint8)
        self.img['bg'] = cv2.dilate(self.img['thresh'],
                                    kernel,
                                    iterations=self.child('bg_iter').value())
        self.img['bg'] = MyFrame(self.img['bg'], 'gray')
        # Use distance transform then threshold to find points
        # within the bounds that could be used as seed
        # for watershed
        self.img['dist'] = cv2.distanceTransform(
            self.img['thresh'], cv2.DIST_L2,
            self.child('dist_iter').value())

        # _, self.img['fg'] = cv2.threshold(
        #     self.img['dist'],
        #     self.child('fg_scale').value() * self.img['dist'].max(), 255, 0)
        # division creates floats, can't have that inside opencv frames
        img_max = np.amax(self.img['dist'])
        if img_max != 0:
            self.img['dist'] = self.img['dist'] * 255 / img_max
        self.img['dist'] = MyFrame(np.uint8(self.img['dist']), 'gray')

        self.img['fg'] = my_threshold(frame=self.img['dist'],
                                      thresh=int(
                                          self.child('fg_scale').value() *
                                          self.img['dist'].max()),
                                      maxval=self.child('adaptive').value(),
                                      type='adaptive')
        if self.img['sel'] is not None:
            if self.img['sel'].shape != self.img['fg'].shape:
                self.img['sel'] = MyFrame(
                    np.zeros(self.img['fg'].shape, dtype=np.uint8))
            self.img['fg'][self.img['sel'] > 0] = 255
        else:
            self.img['sel'] = MyFrame(
                np.zeros(self.img['fg'].shape, dtype=np.uint8))

        # Finding unknown region
        # self.modify_fg_seeds()
        if self.draw_flag:

            self.draw_flag = False

        self.img['fg'] = MyFrame(np.uint8(self.img['fg']), 'gray')
        self.img['unknown'] = MyFrame(
            cv2.subtract(self.img['bg'], self.img['fg']), 'gray')

        # Marker labeling
        # Labels connected components from 0 - n
        # 0 is for background
        count, markers = cv2.connectedComponents(self.img['fg'])
        markers = MyFrame(markers, 'gray')
        print('cc ret:', count)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # print('Markers:', markers)

        # Now, mark the region of unknown with zero
        markers[self.img['unknown'] == 255] = 0
        print('Water:', frame.colorspace)
        markers = cv2.watershed(frame.cvt_color('bgr'), markers)
        # border is -1
        # 0 does not exist
        # bg is 1
        # bubbles is >1
        self.img['final'] = frame.cvt_color('bgr')

        for label in np.unique(markers):
            # if the label is zero, we are examining the 'background'
            # if label is -1, it is the border and we don't need to label it
            # so simply ignore it
            if label == 1 or label == -1:
                continue
            # otherwise, allocate memory
            # for the label region and draw
            # it on the mask
            # print('Label', label)
            mask = np.zeros(self.img['gray'].shape, dtype='uint8')
            mask[markers == label] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(self.img['final'], (int(x), int(y)), int(r),
                       (0, 255, 0), 1)
            if self.child('Overlay', 'Toggle Text').value():
                cv2.putText(self.img['final'], '{}'.format(label),
                            (int(x) - 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 255), 1)

        # self.annotated[markers == -1] = [0, 0, 255]
        # self.annotated[markers == 1] = [0, 0, 0]

        # return unannotated frame for processing

        # overlay selected bubble with fg
        show = self.img['gray'].cvt_color('bgr')
        show[self.img['thresh'] > 0] = (100, 50, 50)
        show[markers == -1] = (0, 0, 255)
        show[self.img['fg'] > 0] = (255, 255, 255)

        if self.show_selection_window:
            if self.prev_window_state == False:  # on change
                cv2.imshow(self.sel_window_title, show)
                cv2.setMouseCallback(self.sel_window_title,
                                     self.on_mouse_click)
            else:  # while showing
                if not cv2.getWindowProperty(self.sel_window_title,
                                             cv2.WND_PROP_VISIBLE):
                    self.show_selection_window = False
                    # cv2.destroyWindow(self.sel_window_title)
                else:
                    cv2.imshow(self.sel_window_title, show)
        self.prev_window_state = self.show_selection_window

        return frame

    # video thread
    def annotate(self, frame):
        view = self.child('Overlay', 'view_list').value()
        ret = MyFrame(cv2.circle(self.img[view], (self.x, self.y), 1,
                           (255, 255, 255), -1))
        return ret

    # main thread
    def on_manual_selection_clicked(self):
        self.show_selection_window = True

    def on_display_mouse_event(self, x, y):
        self.x = x
        self.y = y

    # main thread
    def on_mouse_click(self, event, x, y, flags, params):
        self.event = event
        self.x = x
        self.y = y
        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            self.draw_flag = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("click", event)
            if self.img['sel'] is None:
                self.img['sel'] = np.zeros(self.img['fg'].shape,
                                           dtype=np.uint8)
            self.img['sel'] = MyFrame(
                cv2.circle(self.img['sel'], (self.x, self.y), 1,
                           (255, 255, 255), -1))
            self.draw_flag = True

    def modify_fg_seeds(self):
        # checking for left mouse clicks
        if self.img['sel'] is None:
            self.img['sel'] = np.zeros(self.img['gray'].shape)

        if self.event == cv2.EVENT_LBUTTONDOWN:
            self.img['sel'] = cv2.circle(self.img['sel'], (self.x, self.y), 5,
                                         (255, 255, 255), -1)

        self.img['fg'][self.img['sel'] > 0] = 255